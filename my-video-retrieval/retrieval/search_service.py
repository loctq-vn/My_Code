"""Search service — the single facade the UI and smoke scripts talk to.

Turns a :class:`core.schemas.SearchQuery` into fully enriched
:class:`core.schemas.SearchResult` objects:

* validate / normalize the query (cap ``top_k`` by ``max_top_k``),
* route to the :class:`retrieval.base.BaseRetriever` matching ``query.mode``,
* enrich each ``RawHit`` into a ``SearchResult`` — join the shot row
  (``ShotChunk``) and video metadata (``VideoMeta``) from the in-memory
  :class:`core.metadata_store.MetadataStore`, and build a highlighted snippet.

The service owns none of the storage/model — it only composes the retriever,
the metadata store, and the schemas. Keeps the UI layer thin and the pipeline
swappable (e.g. swap in a FAISS retriever later without touching the UI).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

from config.settings import SETTINGS, Settings
from core.metadata_store import MetadataStore
from core.schemas import SearchMode, SearchQuery, SearchResult, ShotChunk, VideoMeta
from retrieval.base import BaseRetriever, RawHit
from retrieval.lancedb_retriever import LanceDBRetriever
from retrieval.visual_scorer import VisualScorer, fuse_hybrid_visual

# Matches one full Unicode word (Vietnamese ok: letters incl. diacritics).
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class SearchService:
    """Query -> enriched results, with sensible defaults & safety caps."""

    def __init__(
        self,
        settings: Settings = SETTINGS,
        retriever: Optional[BaseRetriever] = None,
        metadata: Optional[MetadataStore] = None,
        visual_scorer: Optional[VisualScorer] = None,
    ) -> None:
        self._settings = settings
        self._metadata = metadata if metadata is not None else MetadataStore(settings)
        self._retriever = retriever if retriever is not None else LanceDBRetriever(
            settings=settings, metadata=self._metadata
        )
        # Lazy SigLIP wrapper: constructing it does NOT load the model; weights
        # are only fetched on the first visual query.
        self._visual = visual_scorer if visual_scorer is not None else VisualScorer(
            settings=settings, metadata=self._metadata
        )

    # ------------------------------------------------------------ helpers -----
    def _normalize(self, query: SearchQuery) -> SearchQuery:
        """Cap ``top_k`` to ``max_top_k`` and default the mode to HYBRID."""
        top_k = max(1, min(query.top_k, self._settings.max_top_k))
        mode = query.mode or SearchMode.HYBRID
        return SearchQuery(
            text=query.text.strip(),
            mode=mode,
            top_k=top_k,
            dataset=query.dataset,
            datasets=query.datasets,
            video_id=query.video_id,
            video_ids=query.video_ids,
            start_time=query.start_time,
            end_time=query.end_time,
            visual_weight=query.visual_weight,
            image_path=query.image_path,
        )

    @staticmethod
    def _snippet(text: str, query_text: str, radius: int = 120) -> str:
        """Return a ``radius``-char window around the first query-term hit.

        Falls back to the text start when no query term is found. Keeps at most
        ``2*radius + len(term)`` characters and elides long tails.
        """
        if not text:
            return ""
        low = text.lower()
        terms = [m.group(0).lower() for m in _TOKEN_RE.finditer(query_text) if len(m.group(0)) >= 3]
        idx = -1
        for term in terms:
            found = low.find(term)
            if found != -1:
                idx = found
                break
        if idx == -1:
            idx = 0
        start = max(0, idx - radius)
        end = min(len(text), idx + radius)
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(text) else ""
        return prefix + text[start:end].replace("\n", " ").strip() + suffix

    # ------------------------------------------------------------- search -----
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Full pipeline: normalize -> retrieve -> enrich -> rank-by-score."""
        q = self._normalize(query)
        if q.mode == SearchMode.VISUAL:
            return self._search_visual(q)
        if q.mode == SearchMode.HYBRID_VISUAL:
            return self._search_hybrid_visual(q)
        raw_hits = self._retriever.search(q)
        results: List[SearchResult] = []
        for position, hit in enumerate(raw_hits, start=1):
            enriched = self._enrich(hit, q, position)
            if enriched is not None:
                results.append(enriched)
        # Retriever already returns best-first; keep a defensive re-sort so the
        # UI can always rely on descending ``score``.
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ------------------------------------------------------ visual modes -----
    @staticmethod
    def _candidate_limit(top_k: int) -> int:
        """Expanded candidate pool for visual re-ranking (mirrors RRF expansion)."""
        return max(top_k * 3, top_k + 50)

    def _search_visual(self, q: SearchQuery) -> List[SearchResult]:
        """Mode VISUAL - SigLIP-only ranking over keyframes (text or image query)."""
        if q.image_path:
            qvec = self._visual.encode_image_query(q.image_path)
        else:
            qvec = self._visual.encode_text_query(q.text)
        ranked = self._visual.store.search_shots(
            qvec,
            dataset=q.dataset,
            video_id=q.video_id,
            video_ids=q.video_ids,
            datasets=q.datasets,
            top_k=q.top_k,
        )
        results: List[SearchResult] = []
        for position, (score, video_id, shot_id) in enumerate(ranked, start=1):
            chunk = self._metadata.get_chunk(video_id, shot_id)
            video = self._metadata.get_video(video_id)
            if chunk is None or video is None:
                continue
            text = chunk.caption or chunk.subtitle or ""
            results.append(
                SearchResult(
                    chunk=chunk,
                    video=video,
                    mode=q.mode,
                    score=float(score),
                    visual_score=float(score),
                    snippet=self._snippet(text, q.text),
                    ranked_position=position,
                )
            )
        return results

    def _search_hybrid_visual(self, q: SearchQuery) -> List[SearchResult]:
        """Mode HYBRID_VISUAL - text-hybrid candidates re-ranked by SigLIP.

        S_final = w * S_rrf_norm + (1 - w) * S_vis_norm with min-max scaling;
        shots without a usable keyframe keep their normalised text score.
        """
        inner = SearchQuery(
            text=q.text,
            mode=SearchMode.HYBRID,
            top_k=self._candidate_limit(q.top_k),
            dataset=q.dataset,
            datasets=q.datasets,
            video_id=q.video_id,
            video_ids=q.video_ids,
            start_time=q.start_time,
            end_time=q.end_time,
        )
        raw_hits = self._retriever.search(inner)

        keys: List[Tuple[str, int]] = []
        rrf_scores: List[float] = []
        for hit in raw_hits:
            parsed = self._parse_document_id(hit.document_id)
            if parsed is None:
                continue
            keys.append(parsed)
            rrf_scores.append(hit.score)
        if not keys:
            return []

        qvec = self._visual.encode_text_query(q.text)
        vis_map = self._visual.score_candidates(qvec, keys)
        vis_scores = [vis_map.get(key, float("nan")) for key in keys]

        final = fuse_hybrid_visual(rrf_scores, vis_scores, weight=q.visual_weight)
        order = np.argsort(-final)[: q.top_k]

        results: List[SearchResult] = []
        for position, idx in enumerate(order, start=1):
            video_id, shot_id = keys[idx]
            chunk = self._metadata.get_chunk(video_id, shot_id)
            video = self._metadata.get_video(video_id)
            if chunk is None or video is None:
                continue
            text = chunk.caption or chunk.subtitle or ""
            vis = vis_scores[idx]
            results.append(
                SearchResult(
                    chunk=chunk,
                    video=video,
                    mode=q.mode,
                    score=float(final[idx]),
                    visual_score=None if np.isnan(vis) else float(vis),
                    snippet=self._snippet(text, q.text),
                    ranked_position=position,
                )
            )
        return results

    # ------------------------------------------------------------- enrich -----
    @staticmethod
    def _parse_document_id(document_id: str) -> Optional[tuple]:
        """Split ``"L25_V001:3"`` -> (video_id, shot_id). Returns None on bad shape."""
        try:
            video_id, shot_id = document_id.rsplit(":", 1)
            return video_id, int(shot_id)
        except (ValueError, TypeError):
            return None

    def _enrich(self, hit: RawHit, query: SearchQuery, position: int) -> Optional[SearchResult]:
        """Attach shot + video metadata to a raw hit, else None if unresolved."""
        parsed = self._parse_document_id(hit.document_id)
        if parsed is None:
            return None
        video_id, shot_id = parsed
        chunk = self._metadata.get_chunk(video_id, shot_id)
        video = self._metadata.get_video(video_id)
        if chunk is None or video is None:
            return None
        text = chunk.caption or chunk.subtitle or ""
        return SearchResult(
            chunk=chunk,
            video=video,
            mode=query.mode,
            score=hit.score,
            vector_score=hit.score if query.mode == SearchMode.VECTOR else None,
            keyword_score=hit.score if query.mode == SearchMode.KEYWORD else None,
            snippet=self._snippet(text, query.text),
            ranked_position=position,
        )