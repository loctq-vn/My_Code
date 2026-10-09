"""In-memory metadata registry for the Video Retrieval UI.

Loads every caption JSON plus the video-titles CSV once at startup into dicts
so lookups by ``(video_id, shot_id)`` and ``video_id`` are O(1). Also exposes
stats, lazy keyframe-path resolution, and a lightweight shot<->embedding
alignment report (header-only reads of the per-video ``.npy`` files).

The store does NOT do search — it is pure metadata plumbing consumed by the
retrieval service and the UI layer.
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from config.settings import SETTINGS, Settings
from core.schemas import ShotChunk, VideoMeta

logger = logging.getLogger(__name__)


class MetadataStore:
    """In-memory registry of shot chunks + resolved video metadata.

    Usage::

        store = MetadataStore()          # loads everything at construction
        chunk = store.get_chunk("L25_V001", 1)
        video = store.get_video("L25_V001")
    """

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self._settings = settings
        self._titles: Dict[str, Tuple[str, Optional[str]]] = {}
        self._chunks: Dict[Tuple[str, int], ShotChunk] = {}
        self._chunks_by_video: Dict[str, List[ShotChunk]] = {}
        self._video_meta: Dict[str, VideoMeta] = {}
        self._video_ids_by_dataset: Dict[str, List[str]] = {}
        self._dataset_roots: Dict[str, Path] = {}
        self._errors: List[str] = []
        self._fps_cache: Dict[str, Optional[float]] = {}
        self.load()

    def load(self) -> None:
        """Load all titles + caption JSONs from disk (call once per store)."""
        self._load_titles()
        for dataset_root in self._settings.dataset_roots():
            self._load_dataset(dataset_root)

    def _load_titles(self) -> None:
        csv_path = self._settings.titles_csv
        if not csv_path.is_file():
            self._errors.append(f"titles CSV missing: {csv_path}")
            return
        try:
            with open(csv_path, encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    vid = (row.get("video_name") or "").strip()
                    if not vid:
                        continue
                    title = (row.get("title") or vid).strip()
                    url = (row.get("youtube_url") or "").strip() or None
                    self._titles[vid] = (title, url)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            self._errors.append(f"titles CSV read failed: {exc}")
            logger.exception("failed to read titles CSV %s", csv_path)

    def _load_dataset(self, dataset_root: Path) -> None:
        dataset = dataset_root.name
        self._dataset_roots[dataset] = dataset_root
        captions_dir = self._settings.captions_dir(dataset_root)
        if not captions_dir.is_dir():
            self._errors.append(f"captions dir missing: {captions_dir}")
            return
        for caption_file in sorted(captions_dir.glob("*.json")):
            self._load_caption_file(dataset, caption_file.stem, caption_file, dataset_root)

    def _load_caption_file(
        self, dataset: str, video_id: str, path: Path, dataset_root: Path
    ) -> None:
        if video_id in self._video_meta:
            self._errors.append(f"duplicate video_id {video_id} (from {path}) - skipped")
            return
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"JSON parse failed {path.name}: {exc}")
            return
        if not isinstance(records, list):
            self._errors.append(f"unexpected JSON shape in {path.name}")
            return

        title, url = self._titles.get(video_id, (video_id, None))
        meta = VideoMeta(
            video_id=video_id,
            title=title,
            youtube_url=url,
            dataset=dataset,
            local_path=self._find_local_video(
                self._settings.videos_dir(dataset_root), video_id
            ),
        )
        self._video_meta[video_id] = meta
        self._video_ids_by_dataset.setdefault(dataset, []).append(video_id)
        self._chunks_by_video[video_id] = []

        added = 0
        for rec in records:
            chunk = self._record_to_chunk(video_id, rec)
            if chunk is None:
                continue
            key = (video_id, chunk.shot_id)
            if key in self._chunks:
                self._errors.append(f"duplicate chunk {key} in {path.name}")
                continue
            self._chunks[key] = chunk
            self._chunks_by_video[video_id].append(chunk)
            added += 1
        self._chunks_by_video[video_id].sort(key=lambda c: c.shot_id)
        logger.info("loaded %d chunks from %s", added, path.name)

    @staticmethod
    def _record_to_chunk(video_id: str, rec: Mapping) -> Optional[ShotChunk]:
        try:
            shot_id = int(rec["shot_id"])
            start_time = float(rec["start_time"])
            end_time = float(rec["end_time"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("bad shot record for %s: %s", video_id, exc)
            return None
        keyframes = tuple(str(k) for k in rec.get("keyframes") or [])
        return ShotChunk(
            video_id=video_id,
            shot_id=shot_id,
            start_time=start_time,
            end_time=end_time,
            duration=float(rec.get("duration") or 0.0),
            subtitle=str(rec.get("subtitle") or ""),
            caption=str(rec.get("caption") or ""),
            memory=str(rec.get("memory") or ""),
            keyframes=keyframes,
        )

    @staticmethod
    def _find_local_video(videos_dir: Path, video_id: str) -> Optional[Path]:
        """Return the local video file path for ``video_id`` (any extension)."""
        if not videos_dir.is_dir():
            return None
        for candidate in videos_dir.glob(f"{video_id}.*"):
            if candidate.is_file():
                return candidate.resolve()
        return None

    # ------------------------------------------------------------ accessors -
    def get_chunk(self, video_id: str, shot_id: int) -> Optional[ShotChunk]:
        """O(1) lookup of a single shot chunk."""
        return self._chunks.get((video_id, int(shot_id)))

    def chunks_for_video(self, video_id: str) -> Tuple[ShotChunk, ...]:
        """All chunks of a video, sorted by ``shot_id`` (empty tuple if unknown)."""
        return tuple(self._chunks_by_video.get(video_id, ()))

    def get_video(self, video_id: str) -> Optional[VideoMeta]:
        """Resolved metadata for a single video (title, URL, local path, dataset)."""
        return self._video_meta.get(video_id)

    def resolve_video(self, video_id: str) -> Optional[VideoMeta]:
        """Alias of :meth:`get_video` (used by the retrieval service)."""
        return self._video_meta.get(video_id)

    def all_video_ids(self) -> Tuple[str, ...]:
        return tuple(self._video_meta.keys())

    def video_ids_for_dataset(self, dataset: str) -> Tuple[str, ...]:
        return tuple(self._video_ids_by_dataset.get(dataset, ()))

    def datasets(self) -> Tuple[str, ...]:
        return tuple(self._dataset_roots.keys())

    def errors(self) -> Tuple[str, ...]:
        return tuple(self._errors)

    def stats(self) -> Mapping[str, object]:
        """Aggregate counts for the UI sidebar / smoke tests."""
        videos_per_dataset = {
            ds: len(ids) for ds, ids in self._video_ids_by_dataset.items()
        }
        chunks_per_dataset = {
            ds: sum(len(self._chunks_by_video.get(v, ())) for v in ids)
            for ds, ids in self._video_ids_by_dataset.items()
        }
        return {
            "datasets": self.datasets(),
            "videos": len(self._video_meta),
            "chunks": len(self._chunks),
            "errors": len(self._errors),
            "videos_per_dataset": videos_per_dataset,
            "chunks_per_dataset": chunks_per_dataset,
        }

    # ------------------------------------------------------------- keyframes -
    def keyframe_path(self, video_id: str, rel_keyframe: str) -> Optional[Path]:
        """Resolve a stored keyframe reference to an absolute existing path.

        JSON stores paths like ``data\\keyframes\\L25_V001\\004.jpg`` while the
        real files live at ``<dataset_root>/keyframes/...`` — the leading
        ``data`` segment is stripped when the direct candidate does not exist.
        """
        meta = self._video_meta.get(video_id)
        if meta is None:
            return None
        root = self._dataset_roots.get(meta.dataset)
        if root is None:
            return None
        rel = Path(str(rel_keyframe).replace("\\", "/"))
        if rel.parts and rel.parts[0] == "data":
            rel = Path(*rel.parts[1:])
        candidate = (root / rel).resolve()
        return candidate if candidate.is_file() else None

    def first_keyframe_path(self, video_id: str, shot_id: int) -> Optional[Path]:
        """Absolute path of the first keyframe of ``shot_id`` (for thumbnails)."""
        chunk = self.get_chunk(video_id, shot_id)
        if chunk is None or not chunk.keyframes:
            return None
        return self.keyframe_path(video_id, chunk.keyframes[0])

    # ------------------------------------------------------- frame mapping ---
    def frame_rate(self, video_id: str) -> Optional[float]:
        """FPS of ``video_id`` from its ``map-keyframes/<video>.csv`` (cached).

        Returns ``None`` when the file is missing or its fps value is invalid,
        so callers can degrade gracefully (e.g. hide the frame range in the UI).
        """
        if video_id in self._fps_cache:
            return self._fps_cache[video_id]

        meta = self._video_meta.get(video_id)
        fps = None
        if meta is not None and meta.dataset in self._dataset_roots:
            csv_path = (
                self._settings.map_keyframes_dir(self._dataset_roots[meta.dataset])
                / f"{video_id}.csv"
            )
            if csv_path.is_file():
                try:
                    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
                        for row in itertools.islice(csv.DictReader(fh), 1):
                            fps = float(row.get("fps") or "nan")
                            if fps <= 0:
                                fps = None
                            break
                except Exception as exc:  # noqa: BLE001 - not fatal for the UI
                    logger.debug("failed to read fps for %s: %s", video_id, exc)
                    fps = None
        self._fps_cache[video_id] = fps
        return fps

    def frame_range(
        self, video_id: str, start_time: float, end_time: float
    ) -> Optional[Tuple[int, int]]:
        """Map a shot's [start, end] seconds to a [start_frame, end_frame] range.

        Uses the video's own fps from ``map-keyframes`` (never assumed constant).
        Returns ``None`` when no reliable fps is available.
        """
        fps = self.frame_rate(video_id)
        if fps is None:
            return None
        return int(round(start_time * fps)), int(round(end_time * fps))

    # ----------------------------------------------------------- alignment -
    def embedding_alignment_report(self) -> Mapping[str, Mapping[str, object]]:
        """Header-only check that per-video .npy embedding rows match shot counts.

        Returns a dict ``{video_id: {expected, actual, aligned, reason?}}``.
        Reads only the NumPy header (mmap), so it is cheap even for large stores.
        """
        import numpy as np

        report: Dict[str, Mapping[str, object]] = {}
        for dataset, root in self._dataset_roots.items():
            emb_dir = self._settings.embeddings_dir(root)
            for video_id in self._video_ids_by_dataset.get(dataset, ()):
                expected = len(self._chunks_by_video.get(video_id, ()))
                npy_file = emb_dir / f"{video_id}.npy"
                if not npy_file.is_file():
                    report[video_id] = {
                        "expected": expected, "actual": None,
                        "aligned": False, "reason": "embedding file missing",
                    }
                    continue
                try:
                    arr = np.load(npy_file, mmap_mode="r")
                    actual = int(arr.shape[0])
                except Exception as exc:  # noqa: BLE001
                    report[video_id] = {
                        "expected": expected, "actual": None,
                        "aligned": False, "reason": str(exc),
                    }
                    continue
                report[video_id] = {
                    "expected": expected, "actual": actual, "aligned": expected == actual,
                }
        return report