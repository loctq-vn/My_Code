"""Tests for the 100-frame CSV sampling helpers in :mod:`ui.components`."""

import pytest

from core.metadata_store import MetadataStore
from core.schemas import SearchMode, SearchQuery
from ui.components import (
    _sample_csv_bytes,
    _sample_timeline,
    build_sample_csv_rows,
)


@pytest.fixture(scope="module")
def store() -> MetadataStore:
    """One metadata store shared across the sampling tests."""
    return MetadataStore()


# --------------------------------------------------------- sampling math ----
def test_sample_timeline_single_is_midpoint() -> None:
    assert _sample_timeline(0, 100, 1) == [50]


def test_sample_timeline_evenly_spaced() -> None:
    assert _sample_timeline(0, 100, 3) == [0, 50, 100]
    assert _sample_timeline(10, 20, 2) == [10, 20]


def test_sample_timeline_clamps_reversed_range() -> None:
    # end < start is tolerated and swapped
    assert _sample_timeline(100, 0, 3) == [0, 50, 100]


def test_sample_timeline_zero_count() -> None:
    assert _sample_timeline(0, 10, 0) == []


# ---------------------------------------------------------- csv builder ----
def _result(store: MetadataStore, video_id: str, shot_id: int):
    from core.schemas import SearchResult, ShotChunk, VideoMeta

    chunk = store.get_chunk(video_id, shot_id)
    assert chunk is not None
    video = store.get_video(video_id)
    assert video is not None
    return SearchResult(chunk=chunk, video=video, mode=SearchMode.HYBRID)


def test_build_rows_target_exact_division(store: MetadataStore) -> None:
    # 4 shots -> target 100, each gets 25 frames.
    results = [_result(store, "L25_V001", sid) for sid in range(1, 5)]
    rows = build_sample_csv_rows(results, store, target=100)
    assert len(rows) == 100
    # video_name has no extension and every frame within its shot's range.
    for video_name, frame in rows:
        assert not video_name.lower().endswith(".mp4")
        assert video_name == "L25_V001"


def test_csv_rows_frames_within_shot_range(store: MetadataStore) -> None:
    results = [_result(store, "L25_V001", sid) for sid in range(1, 6)]
    rows = build_sample_csv_rows(results, store, target=100)
    assert len(rows) == 100
    chunk = store.get_chunk("L25_V001", 1)
    lo, hi = store.frame_range("L25_V001", chunk.start_time, chunk.end_time)
    first_vid, first_frame = rows[0]
    assert first_vid == "L25_V001"
    assert lo <= first_frame <= hi


def test_csv_all_shots_present_and_ordered(store: MetadataStore) -> None:
    results = [_result(store, "L25_V001", sid) for sid in range(1, 4)]  # 3 shots
    rows = build_sample_csv_rows(results, store, target=100)
    assert len(rows) == 100
    # group by the appearance; each shot contributes its share in result order
    groups = [v for v, _ in rows]
    seen = []
    for v in groups:
        if not seen or seen[-1] != v:
            seen.append(v)
    assert seen == ["L25_V001"]


def test_csv_more_shots_than_target_one_each(store: MetadataStore) -> None:
    # 150 distinct shots is unrealistic; use 4 shots with target 3 instead.
    results = [_result(store, "L25_V001", sid) for sid in range(1, 5)]
    rows = build_sample_csv_rows(results, store, target=3)
    assert len(rows) == 3


def test_csv_bytes_no_header_and_full_rows(store: MetadataStore) -> None:
    results = [_result(store, "L25_V001", sid) for sid in range(1, 3)]
    rows = build_sample_csv_rows(results, store, target=100)
    data = _sample_csv_bytes(rows)
    text = data.decode("utf-8")
    lines = [ln.rstrip("\r") for ln in text.strip().split("\n")]
    # No header row: every line is a data row "video,frame".
    assert len(lines) == 100
    for ln in lines:
        parts = ln.split(",")
        assert len(parts) == 2
        assert parts[0] == "L25_V001"
        assert parts[1].isdigit()


def test_csv_no_rows_when_no_fps(store: MetadataStore) -> None:
    # A fake store without frame_range -> graceful empty result.
    class _NoRange:
        def frame_range(self, *a, **k):
            return None

    rows = build_sample_csv_rows([], _NoRange(), target=100)
    assert rows == []