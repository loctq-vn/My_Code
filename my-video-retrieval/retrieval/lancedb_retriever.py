"""LanceDB-backed retriever for the Video Retrieval UI.

Implements :class:`retrieval.base.BaseRetriever` over the index built by
:mod:`ingestion.build_index`:

* ``VECTOR``  — encode the query with ``BAAI/bge-m3`` then ANN search; the
  raw ``_distance`` (L2) from LanceDB is converted to a similarity score via
  ``1 / (1 + distance)``.
* ``KEYWORD`` — Tantivy BM25 over ``subtitle`` + ``caption`` (ascii_folding
  enabled at index time). FTS returns a relevance field that we read
  defensively (``_score`` / ``_distance``) and use as a positive score.
* ``HYBRID``  — runs both retrievers on an expanded candidate set and merges
  them with Reciprocal Rank Fusion (``sum 1/(k + rank)``). LanceDB's built-in
  ``query_type="hybrid"`` requires a registered embedding function, so we do
  the fusion ourselves for full control.

Document ids are the stable ``"{video_id}:{shot_id}"`` strings written by the
index builder; the richer :class:`core.schemas.ShotChunk` / ``VideoMeta``
enrichment happens upstream in :mod:`retrieval.search_service`.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import List, Optional, Tuple

import lancedb

from config.settings import SETTINGS, Settings
from core.metadata_store import MetadataStore
from core.schemas import SearchMode, SearchQuery
from ingestion.embedder import Embedder
from retrieval.base import BaseRetriever, RawHit

logger = logging.getLogger(__name__)

# Unicoded word tokens (Vietnamese letters incl. diacritics OK).
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class LanceDBRetriever(BaseRetriever):
    """Vector / BM25 / hybrid retrieval over a local LanceDB table."""

    # Caption-embedding and subtitle-raw modes are handled here too: caption
    # embedding reuses the vector ANN path; subtitle-raw runs an isolated BM25
    # over the subtitle field (see :meth:`_subtitle_search`).
    supported_modes: Tuple[SearchMode, ...] = (
        SearchMode.VECTOR,
        SearchMode.KEYWORD,
        SearchMode.HYBRID,
        SearchMode.CAPTION_EMBEDDING,
        SearchMode.SUBTITLE_RAW,
    )

    def __init__(
        self,
        settings: Settings = SETTINGS,
        embedder: Optional[Embedder] = None,
        metadata: Optional[MetadataStore] = None,
        table=None,
    ) -> None:
        self._settings = settings
        self._metadata = metadata if metadata is not None else MetadataStore(settings)
        self._embedder = embedder if embedder is not None else Embedder(settings=settings, metadata=self._metadata)
        self._table = table  # optional pre-opened table (tests / warm caches)
        self._db = None

    # ------------------------------------------------------------- plumbing ---
    def _open_table(self):
        if self._table is None:
            if self._db is None:
                self._db = lancedb.connect(self._settings.lancedb_uri)
            self._table = self._db.open_table(self._settings.table_name)
        return self._table

    # ------------------------------------------------------------ filters -----
    def _apply_filters(self, query: SearchQuery) -> Optional[str]:
        """Build a LanceDB ``where`` clause from optional query filters."""
        parts: List[str] = []

        # Merge singular + plural dataset filters into one ``IN (...)` clause.
        datasets = set(query.datasets or ())
        if query.dataset:
            datasets.add(query.dataset)
        if datasets:
            vals = ", ".join(f'"{d}"' for d in sorted(datasets))
            parts.append(f"dataset IN ({vals})")

        # Merge singular + plural video filters into one ``IN (...)` clause.
        video_ids = set(query.video_ids or ())
        if query.video_id:
            video_ids.add(query.video_id)
        if video_ids:
            vals = ", ".join(f'"{v}"' for v in sorted(video_ids))
            parts.append(f"video_id IN ({vals})")

        if query.start_time is not None:
            parts.append(f"start_time >= {query.start_time}")
        if query.end_time is not None:
            parts.append(f"end_time <= {query.end_time}")
        return " AND ".join(parts) if parts else None

    # ------------------------------------------------------------ scoring -----
    @staticmethod
    def _vector_score(distance: float) -> float:
        """Map a (L2) distance to a similarity score in (0, 1]."""
        return 1.0 / (1.0 + max(0.0, distance))

    @staticmethod
    def _fts_score(row: dict) -> float:
        """Read the BM25 relevance from an FTS hit row (version-robust)."""
        score = row.get("_score")
        if score is None:
            score = row.get("_distance", 0.0)  # older behaviour put bm25 here
        return float(score)

    # ------------------------------------------------------------- retrievers -
    def _vector_search(self, query: SearchQuery, limit: Optional[int] = None) -> List[RawHit]:
        table = self._open_table()
        limit = limit or query.top_k
        qvec = self._embedder.encode_query(query.text)
        builder = table.search(qvec).limit(limit)
        where = self._apply_filters(query)
        if where:
            builder = builder.where(where)
        hits: List[RawHit] = []
        for row in builder.to_list():
            distance = float(row.get("_distance", float("inf")))
            hits.append(RawHit(document_id=row["id"], score=self._vector_score(distance)))
        return hits

    def _keyword_search(self, query: SearchQuery, limit: Optional[int] = None) -> List[RawHit]:
        table = self._open_table()
        limit = limit or query.top_k
        builder = table.search(query.text, query_type="fts").limit(limit)
        where = self._apply_filters(query)
        if where:
            builder = builder.where(where)
        hits: List[RawHit] = []
        for row in builder.to_list():
            hits.append(RawHit(document_id=row["id"], score=self._fts_score(row)))
        return hits

    # ------------------------------------------------- subtitle raw only ------
    def _scope_video_ids(self, query: SearchQuery) -> List[str]:
        """Video ids in scope for in-memory scoring, honouring filters.

        Mirrors ``_apply_filters``' dataset/video union+intersection semantics so
        the metadata-side scorer obeys the same scope as the LanceDB path.
        """
        ids = list(self._metadata.all_video_ids())
        video_keep = set(query.video_ids or ())
        if query.video_id:
            video_keep.add(query.video_id)
        if video_keep:
            ids = [v for v in ids if v in video_keep]
        ds_keep = set(query.datasets or ())
        if query.dataset:
            ds_keep.add(query.dataset)
        if ds_keep:
            keep = set()
            for ds in ds_keep:
                keep.update(self._metadata.video_ids_for_dataset(ds))
            ids = [v for v in ids if v in keep]
        return ids

    def _subtitle_search(self, query: SearchQuery, limit: Optional[int] = None) -> List[RawHit]:
        """Lexical BM25 over the raw ``subtitle`` field only (metadata-backed).

        Deliberately independent of the caption-based embeddings and of the
        combined FTS index so it gives a pure "subtitle text" reading. The query
        terms are required to be present in a subtitle to score.
        """
        limit = limit or query.top_k
        terms = [m.group(0).lower() for m in _TOKEN_RE.finditer(query.text)]
        terms = [t for t in terms if len(t) >= 2]
        if not terms:
            return []

        # (document_id, subtitle tokens)
        docs: List[Tuple[str, List[str]]] = []
        for vid in self._scope_video_ids(query):
            for chunk in self._metadata.chunks_for_video(vid):
                if query.start_time is not None and chunk.end_time < query.start_time:
                    continue
                if query.end_time is not None and chunk.start_time > query.end_time:
                    continue
                text = (chunk.subtitle or "").lower()
                toks = [m.group(0) for m in _TOKEN_RE.finditer(text)]
                toks = [t for t in toks if len(t) >= 2]
                if toks:
                    docs.append((f"{vid}:{chunk.shot_id}", toks))

        if not docs:
            return []

        # doc frequency per term, avg length for BM25.
        avgdl = sum(len(toks) for _, toks in docs) / len(docs)
        n_docs = len(docs)
        df: Counter = Counter()
        for _, toks in docs:
            for t in set(toks):
                df[t] += 1

        k1, b = 1.5, 0.75
        scored: List[Tuple[float, str]] = []
        for doc_id, toks in docs:
            tf: Counter = Counter(toks)
            dl = len(toks)
            score = 0.0
            for term in terms:
                if term not in tf:
                    continue
                n = df.get(term, 0)
                idf = math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))
                tf_part = (tf[term] * (k1 + 1.0)) / (
                    tf[term] + k1 * (1.0 - b + b * (dl / avgdl))
                )
                score += idf * tf_part
            if score > 0:
                scored.append((score, doc_id))

        scored.sort(key=lambda x: -x[0])
        return [RawHit(document_id=doc_id, score=s) for s, doc_id in scored[:limit]]

    # ------------------------------------------------------------ fusion -----
    @staticmethod
    def _rrf_fusion(vector_hits: List[RawHit], keyword_hits: List[RawHit], k: int) -> List[RawHit]:
        """Reciprocal Rank Fusion: accumulate ``1 / (k + rank)`` per doc id.

        Higher final score = present at good rank in both lists. Ties broken by
        the accumulated score; stable ordering for determinism.
        """
        rrf: dict = {}
        for rank, hit in enumerate(sorted(vector_hits, key=lambda h: -h.score)):
            rrf[hit.document_id] = rrf.get(hit.document_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, hit in enumerate(sorted(keyword_hits, key=lambda h: -h.score)):
            rrf[hit.document_id] = rrf.get(hit.document_id, 0.0) + 1.0 / (k + rank + 1)
        return [
            RawHit(document_id=doc, score=score)
            for doc, score in sorted(rrf.items(), key=lambda kv: -kv[1])
        ]

    def _hybrid_search(self, query: SearchQuery) -> List[RawHit]:
        """Vector + BM25 over an expanded candidate pool, then RRF + top_k."""
        k = max(1, self._settings.rrf_k)
        candidates = max(query.top_k * 3, query.top_k + 50)
        vec = self._vector_search(query, limit=candidates)
        kw = self._keyword_search(query, limit=candidates)
        fused = self._rrf_fusion(vec, kw, k)
        return fused[: query.top_k]

    # ------------------------------------------------------------- entry -----
    def search(self, query: SearchQuery) -> List[RawHit]:
        """Dispatch on ``query.mode`` and return best-first ``RawHit``s."""
        mode = query.mode or SearchMode.HYBRID
        if not self.supports(mode):
            raise ValueError(f"unsupported search mode: {mode}")
        if mode == SearchMode.VECTOR or mode == SearchMode.CAPTION_EMBEDDING:
            return self._vector_search(query)
        if mode == SearchMode.KEYWORD:
            return self._keyword_search(query)
        if mode == SearchMode.SUBTITLE_RAW:
            return self._subtitle_search(query)
        return self._hybrid_search(query)