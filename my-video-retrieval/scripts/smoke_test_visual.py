"""Headless smoke test for the VISUAL modes (HYBRID_VISUAL + VISUAL).

Downloads/loads SigLIP (google/siglip-so400m-patch14-384, ~2.8 GB) on first
run, then verifies:
  1) HYBRID_VISUAL re-rank changes/augments the text-hybrid ordering,
  2) VISUAL text-only query returns keyframe-ranked shots globally.

Usage (from the app root)::

    python -m scripts.smoke_test_visual
"""

from __future__ import annotations

import sys


def _force_utf8_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass


def main() -> int:
    _force_utf8_stdio()

    from config.settings import SETTINGS
    from core.metadata_store import MetadataStore
    from core.schemas import SearchMode, SearchQuery
    from retrieval import SearchService

    print("=" * 78)
    print("VIDEO RETRIEVAL UI - visual modes smoke test")
    print("=" * 78)

    store = MetadataStore(SETTINGS)
    service = SearchService(settings=SETTINGS, metadata=store)

    # ---- 1) HYBRID_VISUAL: text hybrid + SigLIP re-rank --------------------
    print("\n[1/2] HYBRID_VISUAL (w=0.5)")
    q1 = SearchQuery(
        text="ng\u01b0\u1eddi \u0111\u00e0n b\u00e0 h\u00e0ng ch\u00e0i chi\u1ebfc thuy\u1ec1n ngo\u00e0i xa",
        mode=SearchMode.HYBRID_VISUAL,
        top_k=5,
        visual_weight=0.5,
    )
    res1 = service.search(q1)
    for r in res1:
        vis = "-" if r.visual_score is None else f"{r.visual_score:.4f}"
        print(
            f"  #{r.ranked_position} {r.video_id}:{r.shot_id:>2} "
            f"score={r.score:.4f} vis={vis} | {r.chunk.timestamp_label}"
        )
    assert len(res1) == 5, "expected 5 hybrid-visual hits"

    # compare with pure text hybrid ordering for reference
    base = service.search(SearchQuery(text=q1.text, mode=SearchMode.HYBRID, top_k=5))
    print("  text-only order:", [f"{r.video_id}:{r.shot_id}" for r in base])
    print("  fused     order:", [f"{r.video_id}:{r.shot_id}" for r in res1])

    # ---- 2) VISUAL: text description over all keyframes ---------------------
    print("\n[2/2] VISUAL (text query)")
    q2 = SearchQuery(
        text="a woman in a white lab coat standing in front of a whiteboard",
        mode=SearchMode.VISUAL,
        top_k=5,
    )
    res2 = service.search(q2)
    for r in res2:
        vis = "-" if r.visual_score is None else f"{r.visual_score:.4f}"
        snippet = (r.snippet or "").replace("\n", " ")[:60]
        print(
            f"  #{r.ranked_position} {r.video_id}:{r.shot_id:>2} vis={vis} | {snippet}"
        )
    assert res2, "expected visual hits"

    print("\nVISUAL SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())