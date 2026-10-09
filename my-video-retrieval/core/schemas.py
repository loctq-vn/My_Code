"""Core data types for the Video Retrieval UI.

These are plain immutable records — pure data, no I/O and no search logic —
shared across ingestion, retrieval, and the Streamlit UI. Keeping them here
lets ``metadata_store``, the retrievers, and the UI stay decoupled while
agreeing on one vocabulary.

Usage::

    chunk = ShotChunk(video_id="L25_V001", shot_id=1, start_time=0.0, ...)
    meta  = VideoMeta(video_id="L25_V001", title="...", youtube_url="...", local_path=...)
    hit   = SearchResult(chunk=chunk, video=meta, score=0.92, snippet="...")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple


class SearchMode(str, Enum):
    """Supported retrieval modes exposed to the UI."""

    VECTOR = "vector"  # embedding similarity (bge-m3)
    KEYWORD = "keyword"  # BM25 full-text over subtitle+caption
    HYBRID = "hybrid"  # Reciprocal Rank Fusion of vector + BM25
    HYBRID_VISUAL = "hybrid_visual"  # hybrid + SigLIP keyframe re-rank
    VISUAL = "visual"  # SigLIP-only (text or image query vs keyframes)
    CAPTION_EMBEDDING = "caption_embedding"  # bge-m3 vector over caption embeddings only
    SUBTITLE_RAW = "subtitle_raw"  # lexical BM25 over the raw subtitle text only

    def __str__(self) -> str:  # ergonomic for UI labels
        return self.value


@dataclass(frozen=True)
class ShotChunk:
    """One retrieval unit: a single video shot with all its text + frame refs.

    Field names intentionally mirror the caption JSON keys so ingestion can
    map records 1:1.
    """

    video_id: str  # e.g. "L25_V001"
    shot_id: int  # 1-based shot index within the video
    start_time: float  # seconds (start of shot)
    end_time: float  # seconds (end of shot)
    duration: float = 0.0
    subtitle: str = ""  # Vietnamese ASR subtitle
    caption: str = ""  # VLM-generated description (also used for vector coding)
    memory: str = ""  # structured "#Tag:" metadata block
    keyframes: Tuple[str, ...] = ()  # relative paths as stored in JSON

    # -- display helpers -----------------------------------------------------
    @staticmethod
    def _mmss(seconds: float) -> str:
        """Format seconds as ``MM:SS`` (no hours; minutes can exceed 59)."""
        s = max(0.0, seconds)
        return f"{int(s // 60):02d}:{int(s % 60):02d}"

    @property
    def timestamp_label(self) -> str:
        """Human-readable 'mm:ss' range for the UI."""
        return f"{self._mmss(self.start_time)} \u2013 {self._mmss(self.end_time)}"


@dataclass(frozen=True)
class VideoMeta:
    """Resolved metadata + source paths for one video."""

    video_id: str
    title: str
    youtube_url: Optional[str] = None
    local_path: Optional[Path] = None  # absolute path to the local mp4, if any
    dataset: str = ""  # dataset folder name, e.g. "L25_data"

    @property
    def has_local(self) -> bool:
        return self.local_path is not None and self.local_path.is_file()

    @property
    def youtube_id(self) -> Optional[str]:
        """Extract the 11-char YouTube video id, best-effort."""
        if not self.youtube_url:
            return None
        if "v=" in self.youtube_url:
            return self.youtube_url.split("v=", 1)[1].split("&", 1)[0][:11]
        if "/embed/" in self.youtube_url:
            return self.youtube_url.rsplit("/embed/", 1)[1].split("?", 1)[0][:11]
        return None


@dataclass(frozen=True)
class SearchResult:
    """A single ranked hit returned to the UI, already enriched with video meta."""

    chunk: ShotChunk
    video: VideoMeta
    mode: SearchMode = SearchMode.HYBRID
    score: float = 0.0  # final fused/vector/keyword score (normalized where possible)
    vector_score: Optional[float] = None  # raw bge-m3 cosine (unused in keyword mode)
    keyword_score: Optional[float] = None  # raw BM25 score (unused in vector mode)
    visual_score: Optional[float] = None  # SigLIP keyframe cosine (visual modes)
    snippet: str = ""  # highlighted snippet for the result card
    ranked_position: int = 0  # 1-based overall rank after fusion

    # alias for template-friendly access
    @property
    def video_id(self) -> str:
        return self.chunk.video_id

    @property
    def shot_id(self) -> int:
        return self.chunk.shot_id

    @property
    def start_time(self) -> float:
        return self.chunk.start_time


@dataclass(frozen=True)
class SearchQuery:
    """Normalized input to the retrieval service.

    Carries the query text plus optional filters. The service routes it to the
    matching retriever(s), encodes it when needed, and returns ``SearchResult``s.
    """

    text: str
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = 10  # capped by config.max_top_k inside the service
    dataset: Optional[str] = None  # filter by dataset folder (e.g. "L25_data")
    datasets: Optional[Tuple[str, ...]] = None  # filter by a set of dataset folders
    video_id: Optional[str] = None  # filter to a single video (used for "browse within video")
    video_ids: Optional[Tuple[str, ...]] = None  # restrict search to a set of videos
    start_time: Optional[float] = None  # optional time-range filter (seconds, inclusive)
    end_time: Optional[float] = None  # optional time-range filter (seconds, inclusive)
    visual_weight: float = 0.5  # w in S_final = w*S_rrf + (1-w)*S_vis (HYBRID_VISUAL)
    image_path: Optional[str] = None  # image query source for VISUAL mode

    def __post_init__(self) -> None:
        if not self.text.strip() and not self.image_path:
            raise ValueError("SearchQuery needs non-empty text or an image_path")
        if self.top_k <= 0:
            raise ValueError("SearchQuery.top_k must be positive")
        # Normalize empty collections to None and tuple-ize non-empty ones so the
        # filters below can rely on "None means no restriction".
        for field in ("datasets", "video_ids"):
            value = getattr(self, field)
            if value is not None and len(value) == 0:
                object.__setattr__(self, field, None)
            elif value is not None:
                object.__setattr__(self, field, tuple(dict.fromkeys(value)))
        if self.start_time is not None and self.end_time is not None:
            if self.start_time >= self.end_time:
                raise ValueError("SearchQuery.start_time must be < end_time")
        if not 0.0 <= self.visual_weight <= 1.0:
            raise ValueError("SearchQuery.visual_weight must be within [0, 1]")