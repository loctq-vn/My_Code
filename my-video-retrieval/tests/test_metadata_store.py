"""Tests for the in-memory :class:`core.metadata_store.MetadataStore`.

Run with:  python -m pytest tests -v
"""

import pytest

from core.metadata_store import MetadataStore


@pytest.fixture(scope="module")
def store() -> MetadataStore:
    """One store shared by all tests in this module (load-once semantics)."""
    return MetadataStore()


# ---------------------------------------------------------------- counts -----
def test_l25_v001_chunk_count(store: MetadataStore) -> None:
    chunks = store.chunks_for_video("L25_V001")
    assert len(chunks) == 19  # verified: 19 shots in caption JSON == (19, 1024) .npy
    assert chunks[0].shot_id == 1 and chunks[-1].shot_id == 19


def test_dataset_sizes(store: MetadataStore) -> None:
    assert len(store.video_ids_for_dataset("L25_data")) == 88
    assert len(store.video_ids_for_dataset("L30_data")) == 96


def test_stats_consistency(store: MetadataStore) -> None:
    st = store.stats()
    assert st["datasets"] == ("L25_data", "L30_data")
    assert st["videos"] == 88 + 96
    assert st["chunks"] >= 2000
    assert sum(st["videos_per_dataset"].values()) == st["videos"]
    assert sum(st["chunks_per_dataset"].values()) == st["chunks"]
    assert st["errors"] == 0


# ------------------------------------------------------------ alignment ------
def test_l25_v001_embedding_alignment(store: MetadataStore) -> None:
    report = store.embedding_alignment_report()
    entry = report["L25_V001"]
    assert entry["expected"] == 19
    assert entry["actual"] == 19
    assert entry["aligned"] is True


def test_embedding_alignment_global(store: MetadataStore) -> None:
    report = store.embedding_alignment_report()
    assert len(report) == 88 + 96  # every video reported
    aligned = [v for v, info in report.items() if info["aligned"]]
    assert len(aligned) == len(report), [v for v, i in report.items() if not i["aligned"]]


# ------------------------------------------------------------- lookups -------
def test_get_chunk_integrity(store: MetadataStore) -> None:
    chunk = store.get_chunk("L25_V001", 1)
    assert chunk is not None
    assert chunk.video_id == "L25_V001"
    assert chunk.start_time == 0.0
    assert chunk.memory
    assert chunk.keyframes
    assert isinstance(chunk.keyframes, tuple)


def test_missing_lookups_return_none(store: MetadataStore) -> None:
    assert store.get_chunk("DOES_NOT_EXIST", 0) is None
    assert store.get_chunk("L25_V001", 9999) is None
    assert store.get_video("DOES_NOT_EXIST") is None
    assert store.resolve_video("DOES_NOT_EXIST") is None


def test_video_ids_unique(store: MetadataStore) -> None:
    ids = store.all_video_ids()
    assert len(ids) == len(set(ids))


# ------------------------------------------------------------- video meta ----
def test_video_meta_resolved(store: MetadataStore) -> None:
    meta = store.get_video("L25_V001")
    assert meta is not None
    assert meta.dataset == "L25_data"
    assert meta.title and meta.title != "L25_V001"  # CSV title resolved
    assert meta.youtube_url and "watch?v=" in meta.youtube_url
    assert meta.has_local is True  # local mp4 exists


# ------------------------------------------------------------- keyframes -----
def test_keyframe_path_resolves(store: MetadataStore) -> None:
    chunk = store.get_chunk("L25_V001", 1)
    path = store.keyframe_path("L25_V001", chunk.keyframes[0])
    assert path is not None and path.is_file()
    # 'data\'-prefixed JSON path must be stripped to the real keyframes dir
    assert path.name == "004.jpg"
    assert "keyframes" in path.parts


def test_first_keyframe_helper(store: MetadataStore) -> None:
    path = store.first_keyframe_path("L25_V001", 1)
    assert path is not None and path.is_file()
    assert store.first_keyframe_path("DOES_NOT_EXIST", 1) is None


# -------------------------------------------------------- frame mapping -----
def test_frame_rate_expected(store: MetadataStore) -> None:
    # L25_V001 -> map-keyframes L25_V001.csv: 25.0 fps
    assert store.frame_rate("L25_V001") == 25.0
    # L25_V005 -> 29.97 (proves fps is per-video, not assumed constant)
    assert abs(store.frame_rate("L25_V005") - 29.97) < 1e-6


def test_frame_rate_missing_video_is_none(store: MetadataStore) -> None:
    assert store.frame_rate("DOES_NOT_EXIST") is None


def test_frame_range_l25_v001_shot1(store: MetadataStore) -> None:
    # shot 1: start 0.0 s, end 25.72 s @ 25 fps -> frames 0..643
    chunk = store.get_chunk("L25_V001", 1)
    fr = store.frame_range("L25_V001", chunk.start_time, chunk.end_time)
    assert fr == (0, 643)


def test_frame_range_uses_per_video_fps(store: MetadataStore) -> None:
    # L25_V005 @ 29.97 fps: 1.0 s -> 30 frames
    fr = store.frame_range("L25_V005", 1.0, 2.0)
    assert fr is not None
    assert fr[0] == round(29.97)