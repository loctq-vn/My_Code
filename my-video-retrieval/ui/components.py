"""Reusable Streamlit UI components for the Video Retrieval UI.

Keeps ``app.py`` thin by moving all rendering (sidebar, search bar, result
cards, playback area) here. These functions call ``streamlit`` directly — they
are the only UI-touching module besides ``app.py``.
"""

from __future__ import annotations

import csv
import io
from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st

from core.schemas import SearchMode, SearchResult
from ui.player import PlaybackSpec, build_playback_spec, youtube_embed_html

# Sidebar mode selector: key -> (label, SearchMode)
UI_MODES = {
    "hybrid": ("Hybrid (Text BM25 + Vector)", SearchMode.HYBRID),
    "hybrid_visual": ("Hybrid + Visual Re-rank", SearchMode.HYBRID_VISUAL),
    "visual": ("Visual-Only (Keyframes)", SearchMode.VISUAL),
    "caption_emb": ("Caption Embedding only (vector)", SearchMode.CAPTION_EMBEDDING),
    "subtitle_raw": ("Subtitle raw text only", SearchMode.SUBTITLE_RAW),
}


def _selectable_video_labels(store, dataset_names: list) -> Dict[str, str]:
    """Map video_id -> 'video_id · title' for the given dataset scope (sorted)."""
    labels: Dict[str, str] = {}
    for ds in dataset_names:
        for vid in store.video_ids_for_dataset(ds):
            meta = store.get_video(vid)
            title = meta.title if meta is not None else vid
            labels[vid] = f"{vid} · {title}"
    return dict(sorted(labels.items()))


def render_header() -> None:
    st.set_page_config(page_title="Video Retrieval UI", page_icon="🎬", layout="wide")
    st.title("🎬 Video Retrieval UI")
    st.caption(
        "Hybrid subtitle + multi-modal search over lecture / news videos "
        "(LanceDB vector + BM25, bge-m3 embeddings)."
    )


def render_sidebar(store, current_query: Optional[str]) -> dict:
    """Show dataset stats + search controls; return a settings dict."""
    with st.sidebar:
        st.header("⚙️ Controls")

        st.subheader("Search mode")
        mode_keys = list(UI_MODES)
        labels = {key: label for key, (label, _) in UI_MODES.items()}
        mode_key = st.radio(
            "Mode",
            options=mode_keys,
            index=0,
            format_func=lambda key: labels[key],
        )
        _, mode = UI_MODES[mode_key]

        top_k = st.slider("Top K", min_value=1, max_value=50, value=10)

        visual_weight = 0.5
        image_file = None
        if mode == SearchMode.HYBRID_VISUAL:
            st.subheader("Visual fusion")
            visual_weight = st.slider(
                "Text weight w (rest = keyframes)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="S_final = w * S_text + (1 - w) * S_visual (min-max scaled)",
            )
        elif mode == SearchMode.VISUAL:
            st.subheader("Image query (optional)")
            image_file = st.file_uploader(
                "Upload a frame to find visually similar shots",
                type=["jpg", "jpeg", "png", "webp"],
            )
            st.caption(
                "No upload yet - the search box above will be used as the "
                "image description."
            )

        st.subheader("Filters")
        datasets = list(store.datasets())
        dataset_sel = st.multiselect(
            "Datasets",
            options=datasets,
            default=datasets,
            help="Restrict search to the selected dataset folders.",
        )

        # Video-level filter — options are scoped to the selected datasets so the
        # list stays manageable (L25: 88, L30: 96). Empty selection = no video
        # restriction (search across all videos in scope).
        scope_datasets = dataset_sel if dataset_sel else list(store.datasets())
        video_labels = _selectable_video_labels(store, scope_datasets)
        with st.expander("🎞 Videos — limit search scope", expanded=False):
            st.caption("Leave empty to search all videos in the selected datasets.")
            video_sel = st.multiselect(
                "Videos",
                options=list(video_labels.keys()),
                format_func=lambda v: video_labels[v],
                default=[],
                placeholder="All videos (no restriction)",
            )
        video_ids = list(video_sel)

        st.subheader("Library stats")
        stats = store.stats()
        st.caption(
            f"Videos: **{stats['videos']}**  ·  Chunks: **{stats['chunks']}**  ·  "
            f"Errors: **{stats['errors']}**"
        )
        for ds in datasets:
            st.caption(f"{ds}: {stats['videos_per_dataset'].get(ds, 0)} videos")

        if current_query:
            st.caption(f"Last query: _{current_query}_")

    return {
        "mode": mode,
        "top_k": top_k,
        "datasets": dataset_sel,
        "video_ids": video_ids,
        "visual_weight": visual_weight,
        "image_file": image_file,
        "mode_key": mode_key,
    }


def render_search_bar() -> Optional[str]:
    """A prominent query input; returns the query text (or None if empty)."""
    q = st.text_input(
        "🔎 Search the library",
        placeholder='e.g. "chiếc thuyền ngoài xa" or "boat woman fishing village"',
    )
    return q.strip() or None


def render_play_button(result: SearchResult, key: str) -> bool:
    """A card button that selects a result for playback; returns True if clicked."""
    if st.button(f"▶ Play {result.chunk.timestamp_label}", key=key, width="stretch"):
        st.session_state["selected_result_id"] = f"{result.video_id}:{result.shot_id}"
        return True
    return False


def render_playback_area(selected: Optional[SearchResult]) -> None:
    """Show the video player for a selected result (local or YouTube)."""
    if selected is None:
        st.info("Select a result to play the video from that timestamp.")
        return
    spec = build_playback_spec(selected)
    st.subheader(
        f"▶ {selected.video.title or selected.video_id} · {selected.chunk.timestamp_label}"
    )
    if spec.kind == "local":
        st.video(spec.local_path, start_time=spec.start_time, end_time=spec.end_time)
    elif spec.kind == "youtube":
        st.components.v1.html(youtube_embed_html(spec), height=380)
    else:
        st.warning("No playable source found for this result.")

    with st.expander("Video metadata"):
        st.write(f"**Video ID:** {selected.video_id} · **Shot:** {selected.shot_id}")
        st.write(f"**Timestamps:** {selected.chunk.timestamp_label}")
        if selected.video.youtube_url:
            st.markdown(f"[Open on YouTube]({selected.video.youtube_url})")
        if selected.video.local_path:
            st.caption(f"Local: `{selected.video.local_path}`")


def render_result_card(result: SearchResult, index: int, metadata_store=None) -> None:
    """One card: keyframe thumb, title, timestamps, snippet, score chip."""
    with st.container(border=True):
        cols = st.columns([1, 3])
        with cols[0]:
            kf = _first_keyframe(result, metadata_store)
            if kf:
                st.image(str(kf), width="stretch")
            else:
                st.caption("(no thumbnail)")
        with cols[1]:
            st.markdown(f"**{index}. {result.video.title or result.video_id}**")
            vis_chip = ""
            if getattr(result, "visual_score", None) is not None:
                vis_chip = f" · vis {result.visual_score:.4f}"
            # Optional [start_frame, end_frame] of the shot within the video.
            frame_chip = ""
            fr = _frame_range_chip(result, metadata_store)
            if fr:
                frame_chip = f" · frames {fr}"
            st.caption(
                f"`{result.video_id}` · shot {result.shot_id} · "
                f"{result.chunk.timestamp_label}{frame_chip} · "
                f"score {result.score:.4f}{vis_chip}"
            )
            st.markdown(f"_{result.snippet or '…'}_")
        c = render_play_button(result, key=f"play-{result.video_id}-{result.shot_id}")


def render_results(
    results: List[SearchResult],
    metadata_store=None,
    target_rows: int = 100,
) -> None:
    """All result cards + a download button for a fixed-size frame sample."""
    if not results:
        st.warning("No results. Try a different query or mode.")
        return
    for idx, result in enumerate(results, start=1):
        render_result_card(result, idx, metadata_store)

    st.divider()
    rows = build_sample_csv_rows(results, metadata_store, target=target_rows)
    st.download_button(
        label=f"⬇ Download {target_rows}-frame CSV ({len(rows)} rows)",
        data=_sample_csv_bytes(rows),
        file_name="video_frames_sample.csv",
        mime="text/csv",
        disabled=not rows,
        help="Sample up to N frames evenly across the current result shots.",
    )


def _sample_timeline(start: int, end: int, count: int) -> List[int]:
    """Pick ``count`` evenly spaced integer frames within [start, end] linearly.

    ``count == 1`` -> the midpoint. ``count >= 2`` -> evenly spaced from ``start``
    to ``end`` (inclusive of both when ``count > 1``), each within the range.
    """
    if count <= 0:
        return []
    if end < start:
        start, end = end, start
    if count == 1:
        return [(start + end) // 2]
    # For multiple samples, interpolate linearly; clamp to the integer range.
    return [int(round(start + (i * (end - start)) / (count - 1))) for i in range(count)]


def build_sample_csv_rows(
    results: Sequence[SearchResult],
    metadata_store,
    target: int = 100,
) -> List[Tuple[str, int]]:
    """Evenly distribute ``target`` (video_name, frame_id) rows across the shots.

    - Uses the shot's [start_frame, end_frame] from ``metadata_store.frame_range``
      (video id has no ``.mp4``; edges degrade gracefully when fps is unknown).
    - If more usable shots than ``target``: one frame per shot, first ``target``.
    - Otherwise: the fraction remainder is spread across the first shots so the
      total is exactly ``target``.
    """
    if target <= 0:
        return []

    ranges: List[Tuple[str, int, int]] = []
    for result in results:
        video_id = result.video_id
        fr = metadata_store.frame_range(
            video_id, result.chunk.start_time, result.chunk.end_time
        )
        if fr is None:
            continue
        start, end = fr
        if end <= start:
            continue
        ranges.append((video_id, start, end))

    if not ranges:
        return []

    n = len(ranges)
    if n >= target:
        rows = [
            (video_id, (start + end) // 2) for (video_id, start, end) in ranges[:target]
        ]
        return rows[:target]

    base = target // n
    rem = target % n
    rows: List[Tuple[str, int]] = []
    for i, (video_id, start, end) in enumerate(ranges):
        count = base + (1 if i < rem else 0)
        for frame in _sample_timeline(start, end, count):
            rows.append((video_id, frame))
    return rows[:target]


def _sample_csv_bytes(rows: Sequence[Tuple[str, int]]) -> bytes:
    """Serialize ``(video_name, frame_id)`` rows to CSV bytes (no header)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    for video_name, frame_id in rows:
        writer.writerow([video_name, int(frame_id)])
    return buf.getvalue().encode("utf-8")


def render_empty_state() -> None:
    st.info("Enter a query above to start searching the library.")


def _first_keyframe(result: SearchResult, metadata_store):
    """Resolved keyframe path for the result (via the metadata store), or None."""
    if metadata_store is None:
        return None
    return metadata_store.first_keyframe_path(result.video_id, result.shot_id)


def _frame_range_chip(result: SearchResult, metadata_store) -> Optional[str]:
    """Human-readable ``start–end`` frame range for a shot, or None if unknown."""
    if metadata_store is None or getattr(metadata_store, "frame_range", None) is None:
        return None
    fr = metadata_store.frame_range(
        result.video_id, result.chunk.start_time, result.chunk.end_time
    )
    if fr is None:
        return None
    start, end = fr
    return f"{start}–{end}"