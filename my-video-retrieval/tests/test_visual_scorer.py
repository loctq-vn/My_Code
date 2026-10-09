"""Tests for :mod:`retrieval.visual_scorer` (no SigLIP download required).

The model itself is exercised separately in a manual smoke; here we verify the
keyframe mapping, shot grouping, global search, and the fusion math using the
real on-disk embeddings.

Run with:  python -m pytest tests -v
"""

import numpy as np
import pytest

from config.settings import SETTINGS
from core.metadata_store import MetadataStore
from core.schemas import ShotChunk
from retrieval.visual_scorer import (
    KeyframeVectorStore,
    VisualScorer,
    fuse_hybrid_visual,
    minmax_normalize,
)


@pytest.fixture(scope="module")
def metadata() -> MetadataStore:
    return MetadataStore()


@pytest.fixture(scope="module")
def store(metadata: MetadataStore) -> KeyframeVectorStore:
    return KeyframeVectorStore(settings=SETTINGS, metadata=metadata)


# ------------------------------------------------------------ mapping -------
def test_index_mapping_l25_v001(store: KeyframeVectorStore) -> None:
    idx = store.index_for("L25_V001")
    assert idx is not None
    assert len(idx) == 438
    assert idx[1] == 0 and idx[4] == 3 and idx[438] == 437


def test_matrix_shape_and_norm(store: KeyframeVectorStore) -> None:
    matrix = store.matrix_for("L25_V001")
    assert matrix is not None and matrix.shape == (438, 1152)
    vec = store.vector_for("L25_V001", 4)
    assert vec is not None and vec.shape == (1152,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4


def test_unknown_video_returns_none(store: KeyframeVectorStore) -> None:
    assert store.index_for("DOES_NOT_EXIST") is None
    assert store.matrix_for("DOES_NOT_EXIST") is None
    assert store.shot_rows_for("DOES_NOT_EXIST") is None


# ------------------------------------------------------- shot grouping ------
def test_shot_rows_l25_v001(store: KeyframeVectorStore) -> None:
    groups = store.shot_rows_for("L25_V001")
    assert groups is not None and len(groups) == 19
    first_sid, first_rows = groups[0]
    assert first_sid == 1 and first_rows.tolist() == [3]  # 004.jpg -> row 3


def test_score_shots_self_query(store: KeyframeVectorStore) -> None:
    """Using keyframe 004.jpg's own vector as query must score its shot ~1.0."""
    qvec = store.vector_for("L25_V001", 4)
    scores = store.score_shots_for_video(qvec, "L25_V001")
    assert scores is not None and len(scores) == 19
    assert not np.isnan(scores[0])
    assert abs(scores[0] - 1.0) < 1e-5  # cosine of a vector with itself


def test_global_search_finds_own_shot(store: KeyframeVectorStore) -> None:
    qvec = store.vector_for("L25_V001", 4)
    results = store.search_shots(qvec, top_k=5)
    assert len(results) == 5
    top_score, top_vid, top_sid = results[0]
    assert (top_vid, top_sid) == ("L25_V001", 1)
    assert top_score > 0.999


# ------------------------------------------------------------- facade -------
def test_score_candidates_nan_and_hit(
    store: KeyframeVectorStore, metadata: MetadataStore
) -> None:
    scorer = VisualScorer(metadata=metadata, store=store)
    qvec = store.vector_for("L25_V001", 4)
    out = scorer.score_candidates(qvec, [("L25_V001", 1), ("L25_V001", 2)])
    assert abs(out[("L25_V001", 1)] - 1.0) < 1e-5
    assert isinstance(out[("L25_V001", 2)], float)


# --------------------------------------------------------- fusion math ------
def test_minmax_basic_and_constant() -> None:
    assert np.allclose(minmax_normalize(np.array([0.0, 5.0])), [0.0, 1.0])
    assert np.allclose(minmax_normalize(np.array([7.0, 7.0])), [0.0, 0.0])


def test_fuse_weighted_blend_with_missing() -> None:
    rrf = np.array([3.0, 1.0, 2.0])          # norm -> [1, 0, 0.5]
    vis = np.array([np.nan, 0.2, 0.8])       # masked min-max -> [_, 0, 1]
    final = fuse_hybrid_visual(rrf, vis, weight=0.5)
    assert np.allclose(final, [1.0, 0.0, 0.75])


def test_fuse_all_missing_keeps_text_order() -> None:
    rrf = np.array([3.0, 1.0, 2.0])
    vis = np.array([np.nan, np.nan, np.nan])
    final = fuse_hybrid_visual(rrf, vis, weight=0.5)
    assert np.allclose(final, minmax_normalize(rrf))


def test_fuse_constant_visual_scores() -> None:
    rrf = np.array([1.0, 2.0, 3.0])           # norm -> [0, .5, 1]
    vis = np.array([0.9, 0.9, 0.9])           # constant among scored -> zeros
    final = fuse_hybrid_visual(rrf, vis, weight=0.5)
    assert np.allclose(final, [0.0, 0.25, 0.5])


def test_fuse_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        fuse_hybrid_visual(np.zeros(3), np.zeros(2), weight=0.5)