"""Embedding utilities for the Video Retrieval UI.

Wraps the same sentence-transformers model used at offline indexing time
(``BAAI/bge-m3``, dim 1024) so query encoding is fully consistent with the
pre-computed per-shot caption embeddings stored as ``caption_embeddings/*.npy``.

Consistency contract (matches ``src/making_caption_embedding.ipynb``):
* per-shot text is ``caption`` with fallback to ``subtitle`` (stripped),
* ``model.encode(..., convert_to_numpy=True, normalize_embeddings=True)``,
* output cast to ``float32``.

Query text is encoded **without** any instruction prefix — the stored vectors
were also produced from raw caption text, so adding a bge-m3 retrieval prompt
here would break the metric space.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from config.settings import SETTINGS, Settings
from core.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class Embedder:
    """Lazy, process-wide encoder + loader of pre-computed shot embeddings.

    The heavy SentenceTransformer is only loaded on first use (property
    :attr:`model`), which keeps import time and memory low until the first
    query or index build.
    """

    def __init__(
        self,
        settings: Settings = SETTINGS,
        device: Optional[str] = None,
        metadata: Optional[MetadataStore] = None,
    ) -> None:
        self._settings = settings
        self._device = device  # None -> auto (cuda if available else cpu)
        self._metadata = metadata if metadata is not None else MetadataStore()
        self._model = None
        self._model_dim: Optional[int] = None

    # ------------------------------------------------------------------ model -
    @property
    def model(self):
        """Lazily load + eval the sentence-transformers model (cached)."""
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = self._device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading %s on %s ...", self._settings.embed_model, device)
            self._model = SentenceTransformer(self._settings.embed_model, device=device)
            self._model.eval()
            try:  # sentence-transformers >= 6 renamed the accessor
                self._model_dim = self._model.get_embedding_dimension()
            except AttributeError:  # pragma: no cover - older releases
                self._model_dim = self._model.get_sentence_embedding_dimension()  # type: ignore[attr-defined]
            if self._model_dim != self._settings.embed_dim:
                logger.warning(
                    "model dimension %d differs from configured VRI_EMBED_DIM=%d",
                    self._model_dim, self._settings.embed_dim,
                )
        return self._model

    @property
    def embed_dim(self) -> int:
        return self._settings.embed_dim

    # -------------------------------------------------------------- encoding ---
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of raw texts, normalized float32 (N, dim). Empty->(0, dim)."""
        if not texts:
            return np.empty((0, self._settings.embed_dim), dtype=np.float32)
        processed = [t if (t and t.strip()) else " " for t in texts]
        vecs = self.model.encode(
            processed,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query -> (dim,) float32 (normalized)."""
        return self.encode_texts([text])[0]

    # -------------------------------------------------- shot embeddings -------
    def load_shot_embeddings(
        self, video_id: str, expected: Optional[int] = None, force_recompute: bool = False
    ) -> np.ndarray:
        """Return (N, 1024) of shot embeddings for ``video_id``.

        Prefer the on-disk ``caption_embeddings/{video_id}.npy`` file when its
        shape matches (count over the shots, dim over 1024); otherwise re-encode
        from the in-memory chunks (``caption`` or ``subtitle`` fallback).
        """
        npy_path = self._embeddings_path(video_id)
        if not force_recompute and npy_path is not None and npy_path.is_file():
            arr = np.load(npy_path, mmap_mode="r")
            rows = int(arr.shape[0]); dim = int(arr.shape[1]) if arr.ndim == 2 else 0
            if dim == self._settings.embed_dim and (expected is None or rows == expected):
                return np.asarray(arr, dtype=np.float32)
            logger.warning(
                "alignment mismatch for %s: expected=%s got=(%d,%d) -> re-encode",
                video_id, expected, rows, dim,
            )
        chunks = self._metadata.chunks_for_video(video_id)
        if not chunks:
            raise ValueError(f"no chunks available to embed for {video_id}")
        texts = [(c.caption or c.subtitle or "") for c in chunks]
        return self.encode_texts(texts)

    # ------------------------------------------------------------- helpers ---
    def _embeddings_path(self, video_id: str) -> Optional[Path]:
        """Locate ``caption_embeddings/{video_id}.npy`` through the video's dataset."""
        meta = self._metadata.get_video(video_id)
        if meta is None:
            return None
        root = next((r for r in self._settings.dataset_roots() if r.name == meta.dataset), None)
        if root is None:
            return None
        return self._settings.embeddings_dir(root) / f"{video_id}.npy"