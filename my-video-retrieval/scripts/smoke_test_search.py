"""Headless end-to-end smoke test of the whole search pipeline.

Validates configuration, in-memory metadata, the LanceDB index, query
encoding (bge-m3), and result enrichment — without the Streamlit UI. Prints a
compact summary and returns a process exit code suitable for CI/scripts.

Usage (from the app root)::

    python -m scripts.smoke_test_search

Exits 0 on success, 1 on any failure. Because the first call loads bge-m3 and
opens the index, allow a few seconds on first run.
"""

from __future__ import annotations

import sys

from config.settings import SETTINGS
from core.metadata_store import MetadataStore
from core.schemas import SearchMode, SearchQuery
from retrieval import SearchService

# Sanity queries: mix of Vietnamese keyword / English semantic / hybrid.
QUERIES = [
    ("vi-keyword", "chiếc thuyền ngoài xa", SearchMode.KEYWORD),
    ("en-semantic", "literature analysis boat woman fishing village", SearchMode.VECTOR),
    ("mixed-hybrid", "người đàn bà hàng chài chiếc thuyền ngoài xa nguyễn minh châu", SearchMode.HYBRID),
]


def _format_row(result, idx: int) -> str:
    ts = result.chunk.timestamp_label
    title = (result.video.title or result.video_id)[:42]
    snippet = (result.snippet or "").replace("\n", " ")[:84]
    return f"  {idx:>2}. {result.video_id}:{result.shot_id:>2} | {ts} | score={result.score:.4f} | {title}"


def _force_utf8_stdio() -> None:
    """Windows console (cp1252) cannot encode Vietnamese; force UTF-8 streams."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass


def main() -> int:
    _force_utf8_stdio()
    print("=" * 78)
    print("VIDEO RETRIEVAL UI — headless search smoke test")
    print("=" * 78)

    # 1) configuration + in-memory metadata
    print(f"\n[1/3] config + metadata")
    print(f"  datasets   : {[d.name for d in SETTINGS.dataset_roots()]}")
    print(f"  index uri  : {SETTINGS.lancedb_uri}")
    print(f"  embed model: {SETTINGS.embed_model} (dim {SETTINGS.embed_dim})")
    store = MetadataStore(SETTINGS)
    st = store.stats()
    print(f"  videos={st['videos']}  chunks={st['chunks']}  errors={st['errors']}")
    if st["errors"]:
        print(f"  !! metadata errors: {store.errors()}")
        return 1

    # 2) service + index sanity
    print("\n[2/3] opening search service + index")
    service = SearchService(settings=SETTINGS, metadata=store)
    print("  service ready; table check happens implicitly on first query")

    # 3) run the queries
    print("\n[3/3] running queries")
    failed = 0
    for label, text, mode in QUERIES:
        try:
            results = service.search(SearchQuery(text=text, mode=mode, top_k=5))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] ERROR: {exc!r}")
            failed += 1
            continue
        print(f"\n  [{label}] mode={mode.value} query={text!r} -> {len(results)} hits")
        for i, r in enumerate(results, start=1):
            print(_format_row(r, i))
        if not results:
            print("  !! no hits returned")
            failed += 1

    print("\n" + "=" * 78)
    if failed:
        print(f"SMOKE FAILED with {failed} problem(s)")
        return 1
    print("SMOKE OK — all queries returned enriched results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())