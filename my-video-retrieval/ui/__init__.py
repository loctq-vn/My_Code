"""Streamlit UI layer for the Video Retrieval UI.

Presentation only: :mod:`ui.components` renders search controls + result cards,
:mod:`ui.player` builds the video-player payload. No retrieval logic lives
here — components receive ready :class:`core.schemas.SearchResult` objects.
"""

from . import components, player

__all__ = ["components", "player"]