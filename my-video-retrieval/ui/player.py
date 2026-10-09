"""Video playback helpers for the UI.

Builds a playback payload for a :class:`core.schemas.SearchResult`:

* local mp4  -> stream with ``st.video(path, start_time=sec, end_time=sec)``
  (Streamlit >= 1.41 supports ``start_time``),
* YouTube   -> embed iframe with ``?start=<sec>`` via ``st.components.v1.html``
  (no ``st.video`` YouTube support in current Streamlit).

Pure functions returning tuples so the actual rendering stays thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from core.schemas import SearchResult


@dataclass(frozen=True)
class PlaybackSpec:
    """What the UI should render for one hit."""

    kind: str  # "local" | "youtube" | "none"
    local_path: Optional[str] = None  # absolute mp4 path (kind == "local")
    youtube_url: Optional[str] = None  # full watch url (kind == "youtube")
    youtube_id: Optional[str] = None  # video id (kind == "youtube")
    start_time: Optional[float] = None  # seconds
    end_time: Optional[float] = None  # seconds (None = to end)


def build_playback_spec(result: SearchResult) -> PlaybackSpec:
    """Choose the best playback source for a result."""
    video = result.video
    start = result.chunk.start_time
    end = result.chunk.end_time

    # 1) local file wins (no network, exact timestamps)
    if video.has_local and video.local_path is not None:
        return PlaybackSpec(
            kind="local",
            local_path=str(video.local_path),
            start_time=start,
            end_time=end,
        )

    # 2) fall back to YouTube
    if video.youtube_url:
        return PlaybackSpec(
            kind="youtube",
            youtube_url=video.youtube_url,
            youtube_id=video.youtube_id,
            start_time=start,
            end_time=end,
        )

    # 3) nothing playable
    return PlaybackSpec(kind="none", start_time=start)


def youtube_embed_html(spec: PlaybackSpec, width: int = 640, height: int = 360) -> str:
    """An autoplay-free YouTube iframe with ``t=<sec>`` start + end (no loop)."""
    if spec.youtube_id is None:
        return "<p>YouTube id unavailable.</p>"
    start_sec = int(spec.start_time or 0)
    src = f"https://www.youtube.com/embed/{spec.youtube_id}?start={start_sec}"
    return (
        f'<iframe width="{width}" height="{height}" src="{src}" '
        'frameborder="0" allow="accelerometer; autoplay; clipboard-write; '
        'encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
    )
