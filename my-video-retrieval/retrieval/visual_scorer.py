"""Visual (keyframe) scoring with SigLIP for the Video Retrieval UI.

Cross-modal scoring between a user query (text OR image) and per-shot keyframe
embeddings pre-computed with ``google/siglip-so400m-patch14-384`` (1152-d).

IMPORTANT: this is a *separate* embedding space from the bge-m3 caption
vectors (1024-d) — never compare across the two.

Layout contract (validated on load): within each video, keyframe files are
``001.jpg .. N.jpg`` in natural order and ``keyframe_embeddings/<video>.npy``
has exactly N rows in that same order, so file stem ``k`` reads row ``k-1``.

Fusion (approved spec):
    S_final = w * S_rrf_norm + (1 - w) * S_vis_norm     (both min-max scaled)
Shots without a usable keyframe keep their text position unchanged:
    S_final = S_rrf_norm
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from config.settings import SETTINGS, Settings
from core.metadata_store import MetadataStore
from core.schemas import ShotChunk

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _natural_key(name: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _extract_features(output, prefer: str):
    """Replicate the offline notebook's extraction order exactly:
    raw tensor -> ``.<prefer>`` (text_embeds/image_embeds) -> pooler_output."""
    import torch

    if isinstance(output, torch.Tensor):
        return output
    embeds = getattr(output, prefer, None)
    if embeds is not None:
        return embeds
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    raise TypeError(f"Unsupported model output type: {type(output)}")


# ---------------------------------------------------------------- fusion ----
def minmax_normalize(values) -> np.ndarray:
    """Scale a finite 1-D array into [0, 1]; an all-equal array maps to zeros."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr.copy()
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def fuse_hybrid_visual(rrf_scores, visual_scores, weight: float) -> np.ndarray:
    """Weighted min-max fusion with missing-keyframe fallback.

    ``visual_scores`` may contain NaN where a shot has no usable keyframe;
    those entries fall back to their normalised text score untouched, so a
    good subtitle/caption is never punished for lacking imagery.
    """
    rrf = np.asarray(rrf_scores, dtype=np.float64)
    vis = np.asarray(visual_scores, dtype=np.float64)
    if rrf.shape != vis.shape or rrf.ndim != 1:
        raise ValueError("rrf_scores and visual_scores must be equal-length 1-D arrays")
    rrf_n = minmax_normalize(rrf)
    final = rrf_n.copy()
    mask = ~np.isnan(vis)
    if mask.any():
        vis_n = np.zeros_like(final)
        vis_n[mask] = minmax_normalize(vis[mask])
        final[mask] = weight * rrf_n[mask] + (1.0 - weight) * vis_n[mask]
    return final


# ------------------------------------------------------------- SigLIP -------
class SiglipEmbedder:
    """Lazy SigLIP wrapper exposing text/image query encoding (1152-d)."""

    def __init__(self, settings: Settings = SETTINGS, device: Optional[str] = None) -> None:
        self._settings = settings
        self._device = device
        self._model = None
        self._processor = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoProcessor

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Loading %s on %s ...", self._settings.siglip_model, device)
            self._processor = AutoProcessor.from_pretrained(self._settings.siglip_model)
            model = AutoModel.from_pretrained(self._settings.siglip_model).to(device)
            model.eval()
            self._model = model
        return self._model, self._processor

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        model, processor = self._ensure()
        inputs = processor(text=list(texts), padding="max_length", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.get_text_features(**inputs)
        features = _extract_features(output, prefer="text_embeds")
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        import torch
        from PIL import Image

        images: List = []
        for path in image_paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        model, processor = self._ensure()
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(model.device)
        with torch.no_grad():
            output = model.get_image_features(pixel_values=pixel_values)
        features = _extract_features(output, prefer="image_embeds")
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def encode_query_text(self, text: str) -> np.ndarray:
        """Single text query -> (dim,) float32 unit vector."""
        return self.encode_texts([text])[0]

    def encode_query_image(self, image_path) -> np.ndarray:
        """Single image query -> (dim,) float32 unit vector."""
        return self.encode_images([image_path])[0]

# --------------------------------------------- keyframe vector store -------
class KeyframeVectorStore:
    """Filename-stem -> embedding-row lookup per video (+ shot grouping).

    Validates the layout contract on first access per video: number of image
    files must match the ``.npy`` row count (mismatches are logged and the
    overlap is used). Keeps per-video memmaps so memory stays flat.
    """

    def __init__(
        self,
        settings: Settings = SETTINGS,
        metadata: Optional[MetadataStore] = None,
    ) -> None:
        self._settings = settings
        self._metadata = metadata if metadata is not None else MetadataStore(settings)
        self._stem_to_row: Dict[str, Dict[int, int]] = {}
        self._arrays: Dict[str, np.ndarray] = {}
        self._shot_rows: Dict[str, List[Tuple[int, np.ndarray]]] = {}

    # -- location -----------------------------------------------------------
    def _paths_for(self, video_id: str) -> Optional[Tuple[Path, Path]]:
        for root in self._settings.dataset_roots():
            kf_dir = self._settings.keyframes_dir(root) / video_id
            emb_path = self._settings.keyframe_embeddings_dir(root) / f"{video_id}.npy"
            if kf_dir.is_dir() and emb_path.is_file():
                return kf_dir, emb_path
        return None

    def index_for(self, video_id: str) -> Optional[Dict[int, int]]:
        """``{file stem int: npy row}`` for a video (built lazily, cached)."""
        cached = self._stem_to_row.get(video_id)
        if cached is not None:
            return cached
        located = self._paths_for(video_id)
        if located is None:
            return None
        kf_dir, emb_path = located
        files = [p for p in kf_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
        files.sort(key=lambda p: _natural_key(p.name))
        array = np.load(emb_path, mmap_mode="r")
        n_rows = int(array.shape[0])
        if n_rows != len(files):
            logger.warning(
                "keyframe count mismatch for %s (%d files vs %d rows) - using overlap",
                video_id, len(files), n_rows,
            )
        mapping: Dict[int, int] = {}
        for row in range(min(len(files), n_rows)):
            stem = files[row].stem
            if stem.isdigit():
                mapping[int(stem)] = row
        self._stem_to_row[video_id] = mapping
        self._arrays[video_id] = array
        return mapping

    def matrix_for(self, video_id: str) -> Optional[np.ndarray]:
        """The (N, 1152) float32 memmap for a video, or None if unavailable."""
        if self.index_for(video_id) is None:
            return None
        return self._arrays[video_id]

    def vector_for(self, video_id: str, frame_stem: int) -> Optional[np.ndarray]:
        """One keyframe vector by filename stem (e.g. 4 for ``004.jpg``)."""
        mapping = self.index_for(video_id)
        if not mapping or frame_stem not in mapping:
            return None
        return np.asarray(self._arrays[video_id][mapping[frame_stem]], dtype=np.float32)

    # -- shot grouping --------------------------------------------------------
    @staticmethod
    def _stems_from_chunk(chunk: ShotChunk) -> List[int]:
        stems: List[int] = []
        for ref in chunk.keyframes or ():
            stem = Path(str(ref)).stem
            if stem.isdigit():
                stems.append(int(stem))
        return stems

    def shot_rows_for(self, video_id: str) -> Optional[List[Tuple[int, np.ndarray]]]:
        """``[(shot_id, rows)]`` sorted by shot_id; empty arrays when unmapped."""
        cached = self._shot_rows.get(video_id)
        if cached is not None:
            return cached
        mapping = self.index_for(video_id)
        if mapping is None:
            return None
        groups: List[Tuple[int, np.ndarray]] = []
        for chunk in self._metadata.chunks_for_video(video_id):
            rows = np.array(
                [mapping[s] for s in self._stems_from_chunk(chunk) if s in mapping],
                dtype=np.int64,
            )
            groups.append((chunk.shot_id, rows))
        self._shot_rows[video_id] = groups
        return groups

    # -- scoring ----------------------------------------------------------------
    def score_shots_for_video(self, query_vec, video_id: str) -> Optional[np.ndarray]:
        """Per-shot max keyframe cosine, aligned with ``chunks_for_video`` order.

        Shots without any mapped keyframe get NaN (handled by the fusion).
        """
        groups = self.shot_rows_for(video_id)
        matrix = self.matrix_for(video_id)
        if groups is None or matrix is None:
            return None
        sims = matrix @ np.asarray(query_vec, dtype=np.float32)
        scores = np.full(len(groups), np.nan, dtype=np.float64)
        for i, (_shot_id, rows) in enumerate(groups):
            if rows.size:
                scores[i] = float(np.max(sims[rows]))
        return scores

    def iter_videos(
        self,
        dataset: Optional[str] = None,
        video_id: Optional[str] = None,
        video_ids: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
    ) -> Sequence[str]:
        """Video ids to scan, honouring optional dataset / video filters.

        All scopes are *intersected*: ``video_id`` (single) wins over
        ``video_ids``; each is intersected with the dataset scope when both are
        supplied. Returns the video ids in the store's natural order.
        """
        if video_id:
            return [video_id]

        scoped = list(self._metadata.all_video_ids())

        if video_ids:
            keep = set(video_ids)
            scoped = [v for v in scoped if v in keep]

        # Merge singular + plural dataset filters.
        ds_names = set(datasets or ())
        if dataset:
            ds_names.add(dataset)
        if ds_names:
            keep = set()
            for ds in ds_names:
                keep.update(self._metadata.video_ids_for_dataset(ds))
            scoped = [v for v in scoped if v in keep]

        return scoped

    def search_shots(self, query_vec, dataset=None, video_id=None, video_ids=None, datasets=None, top_k=50):
        """Global visual-only search -> ranked ``(score, video_id, shot_id)``."""
        results: List[Tuple[float, str, int]] = []
        for vid in self.iter_videos(
            dataset=dataset, video_id=video_id, video_ids=video_ids, datasets=datasets
        ):
            scores = self.score_shots_for_video(query_vec, vid)
            if scores is None:
                continue
            chunks = self._metadata.chunks_for_video(vid)
            for i, sc in enumerate(scores):
                if not np.isnan(sc):
                    results.append((float(sc), vid, chunks[i].shot_id))
        results.sort(key=lambda t: -t[0])
        return results[:top_k]


# --------------------------------------------------------------- facade -----
class VisualScorer:
    """Facade combining the SigLIP encoder and the keyframe vector store."""

    def __init__(
        self,
        settings: Settings = SETTINGS,
        metadata: Optional[MetadataStore] = None,
        embedder: Optional[SiglipEmbedder] = None,
        store: Optional[KeyframeVectorStore] = None,
    ) -> None:
        self._settings = settings
        self._metadata = metadata if metadata is not None else MetadataStore(settings)
        self._embedder = embedder if embedder is not None else SiglipEmbedder(settings=settings)
        self._store = (
            store
            if store is not None
            else KeyframeVectorStore(settings=settings, metadata=self._metadata)
        )

    @property
    def embedder(self) -> SiglipEmbedder:
        return self._embedder

    @property
    def store(self) -> KeyframeVectorStore:
        return self._store

    # -- query encoding (loads SigLIP on first use) ---------------------------
    def encode_text_query(self, text: str) -> np.ndarray:
        return self._embedder.encode_query_text(text)

    def encode_image_query(self, image_path) -> np.ndarray:
        return self._embedder.encode_query_image(image_path)

    # -- re-rank support (model-free once the query vector exists) ------------
    def score_candidates(
        self, query_vec, candidates: Sequence[Tuple[str, int]]
    ) -> Dict[Tuple[str, int], float]:
        """Visual cosine per ``(video_id, shot_id)``; NaN where no keyframe."""
        q = np.asarray(query_vec, dtype=np.float32)
        by_video: Dict[str, List[int]] = {}
        for video_id, shot_id in candidates:
            by_video.setdefault(video_id, []).append(shot_id)

        out: Dict[Tuple[str, int], float] = {}
        for video_id, shot_ids in by_video.items():
            mapping = self._store.index_for(video_id)
            matrix = self._store.matrix_for(video_id)
            sims = None if matrix is None else matrix @ q
            for shot_id in shot_ids:
                key = (video_id, shot_id)
                if sims is None:
                    out[key] = float("nan")
                    continue
                chunk = self._metadata.get_chunk(video_id, shot_id)
                stems = self._store._stems_from_chunk(chunk) if chunk else []
                rows = [mapping[s] for s in stems if s in mapping]
                out[key] = float(np.max(sims[rows])) if rows else float("nan")
        return out