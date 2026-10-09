"""Central, typed configuration for the Video Retrieval UI.

This module is the single source of truth for:

* where datasets / metadata / videos live (all paths are composed from a
  workspace root so new ``*_data`` datasets are drop-in),
* which embedding model + dimension the index was built with
  (``BAAI/bge-m3``, dim 1024 — must match the offline ``.npy`` files),
* where the LanceDB index is stored,
* search defaults (top-k, RRF constant, BM25 text fields).

Overrides are read from environment variables (prefix ``VRI_``) and an
optional ``.env`` file — app-level ``my-video-retrieval/.env`` wins over the
workspace ``My_Code/.env``. Reading env/.env is the *only* I/O performed here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Tuple

try:  # python-dotenv is required, but keep the import defensive
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv unavailable
    load_dotenv = None

# Package / workspace layout -------------------------------------------------
# settings.py -> <app-root>/config/settings.py ; app-root is the parent of config/
PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT: Final[Path] = PACKAGE_ROOT.parent

_ENV_PREFIX: Final[str] = "VRI_"


def _load_dotenv_files() -> None:
    """Best-effort load of ``.env`` files (first definition wins)."""
    if load_dotenv is None:
        return
    for path in (PACKAGE_ROOT / ".env", WORKSPACE_ROOT / ".env"):
        if path.is_file():
            load_dotenv(dotenv_path=path, override=False)


_load_dotenv_files()


def _env_str(key: str, default: str) -> str:
    return os.environ.get(_ENV_PREFIX + key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(_ENV_PREFIX + key)
    return int(raw) if raw is not None and raw.strip() else default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(_ENV_PREFIX + key)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable app configuration (use the module-level ``SETTINGS`` singleton)."""

    # ------------------------------------------------------------ paths ----
    workspace_root: Path = WORKSPACE_ROOT
    """Root folder holding the ``*_data`` datasets and the titles CSV."""

    dataset_glob: str = _env_str("DATASET_GLOB", "*_data")
    """Glob (relative to ``workspace_root``) matching dataset folders (e.g. L25_data)."""

    captions_subdir: str = _env_str("CAPTIONS_SUBDIR", "captions")
    embeddings_subdir: str = _env_str("EMBEDDINGS_SUBDIR", "caption_embeddings")
    videos_subdir: str = _env_str("VIDEOS_SUBDIR", "videos")
    keyframes_subdir: str = _env_str("KEYFRAMES_SUBDIR", "keyframes")
    keyframe_embeddings_subdir: str = _env_str(
        "KEYFRAME_EMBEDDINGS_SUBDIR", "keyframe_embeddings"
    )
    shots_subdir: str = _env_str("SHOTS_SUBDIR", "shots")
    map_keyframes_subdir: str = _env_str("MAP_KEYFRAMES_SUBDIR", "map-keyframes")

    titles_csv: Path = WORKSPACE_ROOT / "all_video_titles_and_links.csv"
    """CSV mapping video_name -> title + youtube_url (dataset, video_name, title, youtube_url)."""

    # ------------------------------------------------------------ model ----
    embed_model: str = _env_str("EMBED_MODEL", "BAAI/bge-m3")
    """Sentence-transformers model used for query embedding (matches offline .npy)."""

    embed_dim: int = _env_int("EMBED_DIM", 1024)
    """Dimension of stored caption embeddings -> fixed_size_list<float32>[dim]."""

    # -------------------------------------------------- visual (SigLIP) ----
    siglip_model: str = _env_str("SIGLIP_MODEL", "google/siglip-so400m-patch14-384")
    """Cross-modal encoder for keyframe <-> query scoring (image/text, 1152-d)."""

    siglip_dim: int = _env_int("SIGLIP_DIM", 1152)
    """Dimension of the stored per-keyframe embeddings."""

    # ------------------------------------------------------------ index ----
    lancedb_uri: str = _env_str("LANCEDB_URI", str(PACKAGE_ROOT / "data" / "index"))
    """LanceDB local-directory URI. The table is created at <uri>/<table_name>.lance."""

    table_name: str = _env_str("TABLE_NAME", "shots")

    # ----------------------------------------------------------- search ----
    default_top_k: int = _env_int("DEFAULT_TOP_K", 10)
    max_top_k: int = _env_int("MAX_TOP_K", 50)
    rrf_k: int = _env_int("RRF_K", 60)

    text_fields: Tuple[str, ...] = ("subtitle", "caption")
    """Columns used for BM25 full-text search."""

    # ------------------------------------------------------------- fts ----
    fts_tokenizer: str = _env_str("FTS_TOKENIZER", "simple")
    fts_ascii_folding: bool = _env_bool("FTS_ASCII_FOLDING", True)
    fts_language: str = _env_str("FTS_LANGUAGE", "English")

    # ------------------------------------------------------- validation ----
    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError("VRI_EMBED_DIM must be a positive integer")
        if self.default_top_k <= 0 or self.rrf_k <= 0:
            raise ValueError("DEFAULT_TOP_K and RRF_K must be positive")
        if not self.text_fields:
            raise ValueError("at least one text field is required for BM25")

    # --------------------------------------------------------- helpers ----
    @property
    def lancedb_path(self) -> Path:
        """LanceDB storage location as a Path (local directory backend)."""
        return Path(self.lancedb_uri)

    def dataset_roots(self) -> Tuple[Path, ...]:
        """Existing dataset folders resolved from ``workspace_root`` + glob."""
        return tuple(
            p for p in sorted(self.workspace_root.glob(self.dataset_glob)) if p.is_dir()
        )

    # --- per-dataset subdirectory helpers ----------------------------------
    def captions_dir(self, dataset_root: Path) -> Path:
        return dataset_root / self.captions_subdir

    def embeddings_dir(self, dataset_root: Path) -> Path:
        return dataset_root / self.embeddings_subdir

    def videos_dir(self, dataset_root: Path) -> Path:
        return dataset_root / self.videos_subdir

    def keyframes_dir(self, dataset_root: Path) -> Path:
        return dataset_root / self.keyframes_subdir

    def keyframe_embeddings_dir(self, dataset_root: Path) -> Path:
        return dataset_root / self.keyframe_embeddings_subdir

    def shots_dir(self, dataset_root: Path) -> Path:
        return dataset_root / self.shots_subdir

    def map_keyframes_dir(self, dataset_root: Path) -> Path:
        return dataset_root / self.map_keyframes_subdir


SETTINGS: Final[Settings] = Settings()
"""Immutable process-wide configuration (import this everywhere)."""