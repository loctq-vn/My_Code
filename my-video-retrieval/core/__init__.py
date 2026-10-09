"""Domain layer for the Video Retrieval UI.

Pure data types (:mod:`core.schemas`) shared across config, ingestion,
retrieval, and the UI layer — no I/O, no search logic.
"""

from .schemas import (
    SearchMode,
    SearchQuery,
    SearchResult,
    ShotChunk,
    VideoMeta,
)

__all__ = ["SearchMode", "SearchQuery", "SearchResult", "ShotChunk", "VideoMeta"]