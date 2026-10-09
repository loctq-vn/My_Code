"""Build / rebuild the LanceDB search index for the Video Retrieval UI.

Walks every dataset folder (``L25_data``, ``L30_data``, ...) discovered via
:class:`config.settings.Settings`, maps each caption JSON to one row per shot
with its ``BAAI/bge-m3`` 1024-dim vector (from ``caption_embeddings/*.npy``),
writes them to a LanceDB table with an explicit Arrow schema, then creates a
BM25 full-text (FTS) index over the text columns.

Idempotent: ``--rebuild`` drops and recreates the table; otherwise rows are
appended. Run from the app root::

    python -m ingestion.build_index [--rebuild]

Gates to verify manually afterwards:
* ``tbl.count_rows() == total shots`` (2883 for L25+L30),
* ``tbl.search(vec).limit(5).to_list()`` returns hits,
* ``tbl.search("chài", query_type="fts").limit(5).to_list()`` returns hits.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

import lancedb
from config.settings import SETTINGS, Settings
from core.metadata_store import MetadataStore
from ingestion.embedder import Embedder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_index")

# Optional FTS config — imported inside main() so a bare import stays light,
# and so users can easily tune tokenizer / folding via CLI flags.
_FTS_CONFIG_MODULE = "lancedb.index"

# NOTE: FTS text columns are a subset of the stored columns.
TEXT_FIELDS: Tuple[str, ...] = ("subtitle", "caption")
INDEX_TEXT_FIELDS: Tuple[str, ...] = TEXT_FIELDS


def build_plan(settings: Settings) -> Dict[str, Dict[str, object]]:
    """Resolve every video -> (caption_json, embeddings_npy, shot_count).

    Returns ``{video_id: {"dataset":..., "caption":Path, "npy":Path|None,
    "expected_shots":int}}`` sorted for a deterministic build order.
    """
    store = MetadataStore(settings)
    plan: Dict[str, Dict[str, object]] = {}
    for dataset_root in settings.dataset_roots():
        dataset = dataset_root.name
        captions_dir = settings.captions_dir(dataset_root)
        if not captions_dir.is_dir():
            continue
        for caption_file in sorted(captions_dir.glob("*.json")):
            video_id = caption_file.stem
            shots = store.chunks_for_video(video_id)
            npy = settings.embeddings_dir(dataset_root) / f"{video_id}.npy"
            plan[video_id] = {
                "dataset": dataset,
                "caption": caption_file,
                "npy": npy if npy.is_file() else None,
                "expected_shots": len(shots),
            }
    return plan


def iter_rows(settings: Settings, plan: Dict[str, Dict[str, object]]) -> Iterator[Dict[str, object]]:
    """Yield one row dict per shot, in stable (dataset, video, shot) order.

    ``id`` column is a stable row ID ``"{video_id}:{shot_id}"``. The vector is
    copied from the per-video ``.npy`` when the file exists and its row count
    matches ``expected_shots``; otherwise re-encoded on the fly — but then we
    log a clear warning so a broken index is never silently built.
    """
    store = MetadataStore(settings)
    embedder = Embedder(settings=settings, metadata=store)
    for video_id in sorted(plan):
        info = plan[video_id]
        dataset, caption, npy, expected = (
            info["dataset"], info["caption"], info["npy"], info["expected_shots"],
        )
        vectors: np.ndarray
        if npy is not None:
            arr = np.load(npy, mmap_mode="r")
            if arr.ndim == 2 and arr.shape[1] == settings.embed_dim and arr.shape[0] == expected:
                vectors = np.asarray(arr, dtype=np.float32)
            else:
                logger.warning(
                    "shape mismatch %s: npy=%s expected=%s -> re-encode",
                    video_id, arr.shape, expected,
                )
                vectors = embedder.load_shot_embeddings(video_id, expected=expected, force_recompute=True)
        else:
            logger.warning("missing npy for %s -> re-encode", video_id)
            vectors = embedder.load_shot_embeddings(video_id, expected=expected, force_recompute=True)

        chunks = store.chunks_for_video(video_id)
        if len(chunks) != expected or len(chunks) != vectors.shape[0]:
            logger.error("ALIGNMENT FAIL for %s: shots=%d vecs=%d", video_id, len(chunks), vectors.shape[0])
            raise RuntimeError(f"shot/vector alignment broken for {video_id}")

        for i, chunk in enumerate(chunks):
            yield {
                "id": f"{video_id}:{chunk.shot_id}",
                "video_id": video_id,
                "shot_id": chunk.shot_id,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "dataset": dataset,
                "subtitle": chunk.subtitle,
                "caption": chunk.caption,
                "memory": chunk.memory,
                "vector": vectors[i],
            }


def build_schema(settings: Settings):
    """LanceDB Arrow schema: id/text cols + fixed_size_list<float32>[embed_dim]."""
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("video_id", pa.string()),
            pa.field("shot_id", pa.int64()),
            pa.field("start_time", pa.float64()),
            pa.field("end_time", pa.float64()),
            pa.field("dataset", pa.string()),
            pa.field("subtitle", pa.string()),
            pa.field("caption", pa.string()),
            pa.field("memory", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), settings.embed_dim)),
        ]
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build/rebuild the LanceDB search index.")
    parser.add_argument("--rebuild", action="store_true", help="drop and recreate the table")
    args = parser.parse_args(argv)

    settings = SETTINGS
    plan = build_plan(settings)
    total_videos = len(plan)
    total_shots = sum(int(p["expected_shots"]) for p in plan.values())
    logger.info("plan: %d videos, %d shots across %s", total_videos, total_shots, [d.name for d in settings.dataset_roots()])

    db = lancedb.connect(settings.lancedb_uri)
    tbl = db.create_table(
        settings.table_name,
        schema=build_schema(settings),
        mode="overwrite" if args.rebuild else "create",
        exist_ok=True,
    )
    if args.rebuild:
        logger.info("recreated table %s (mode=overwrite)", settings.table_name)

    from lancedb.index import FTS

    t0 = time.perf_counter()
    batch: List[Dict[str, object]] = []
    inserted = 0
    for row in iter_rows(settings, plan):
        batch.append(row)
        if len(batch) >= 512:
            tbl.add(batch)
            inserted += len(batch)
            batch = []
            logger.info("inserted %d rows ...", inserted)
    if batch:
        tbl.add(batch)
        inserted += len(batch)
    dt = time.perf_counter() - t0
    logger.info("inserted %d rows in %.1fs (%.1f rows/s)", inserted, dt, inserted / dt if dt else 0)

    for text_field in INDEX_TEXT_FIELDS:
        tbl.create_index(text_field, config=FTS(
            with_position=True,
            base_tokenizer=settings.fts_tokenizer,
            ascii_folding=settings.fts_ascii_folding,
            language=settings.fts_language,
        ))
        logger.info("created FTS index on %r", text_field)

    n = tbl.count_rows()
    logger.info("table rows=%d (expected %d)  dim=%d vectors=%s", n, total_shots, settings.embed_dim, settings.lancedb_uri)
    if n != total_shots:
        logger.error("ROW COUNT MISMATCH: %d != %d", n, total_shots)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())