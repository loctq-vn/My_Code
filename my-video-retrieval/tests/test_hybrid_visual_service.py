"""Model-free tests for HYBRID_VISUAL fusion inside :class:`SearchService`.

Uses a fake retriever (fixed text scores) and a fake visual scorer (fixed
SigLIP cosines incl. a missing-keyframe NaN) so the fusion math and fallback
behaviour are verified without loading any neural network.

Run with:  python -m pytest tests -v
"""

import numpy as np
import pytest

from config.settings import SETTINGS
from core.metadata_store import MetadataStore
from core.schemas import SearchMode, SearchQuery
from retrieval.base import BaseRetriever, RawHit
from retrieval.search_service import SearchService


class FakeRetriever(BaseRetriever):
    """Returns fixed text-hybrid scores, honouring top_k."""

    def __init__(self, docs):
        # docs: list of (document_id, score), best first
        self.docs = docs

    def search(self, query):
        return [RawHit(document_id=d, score=s) for d, s in self.docs[: query.top_k]]


class FakeScorer:
    """Duck-typed VisualScorer: deterministic cosines, missing -> NaN."""

    def __init__(self, table):
        self.table = table  # {(video_id, shot_id): cosine}

    def encode_text_query(self, text):
        return np.ones(4, dtype=np.float32)

    def score_candidates(self, qvec, candidates):
        return {key: self.table.get(key, float("nan")) for key in candidates}


@pytest.fixture(scope="module")
def metadata() -> MetadataStore:
    return MetadataStore()


def _make_service(metadata: MetadataStore) -> SearchService:
    retriever = FakeRetriever(
        [
            ("L25_V001:1", 0.9),
            ("L25_V001:2", 0.5),
            ("L25_V001:3", 0.1),
        ]
    )
    scorer = FakeScorer(
        {
            ("L25_V001", 1): float("nan"),  # no usable keyframe -> fallback
            ("L25_V001", 2): 0.5,
            ("L25_V001", 3): 0.95,
        }
    )
    return SearchService(
        settings=SETTINGS,
        metadata=metadata,
        retriever=retriever,
        visual_scorer=scorer,
    )


def _ids(results):
    return [(r.video_id, r.shot_id) for r in results]


def test_hybrid_visual_ranking_and_fallback(metadata: MetadataStore) -> None:
    svc = _make_service(metadata)
    q = SearchQuery(
        text="anything",
        mode=SearchMode.HYBRID_VISUAL,
        top_k=3,
        visual_weight=0.5,
    )
    results = svc.search(q)

    # rrf_norm = [1, .5, 0]; masked vis min-max = [_, 0, 1]
    # final A = 1.0 (fallback), B = .25, C = .5  => order A, C, B
    assert _ids(results) == [("L25_V001", 1), ("L25_V001", 3), ("L25_V001", 2)]
    assert abs(results[0].score - 1.0) < 1e-9
    assert abs(results[1].score - 0.5) < 1e-9
    assert abs(results[2].score - 0.25) < 1e-9

    # missing keyframe keeps text position AND has no visual score attached
    assert results[0].visual_score is None
    assert abs(results[1].visual_score - 0.95) < 1e-9
    assert all(r.mode == SearchMode.HYBRID_VISUAL for r in results)


def test_hybrid_visual_weight_one_is_pure_text(metadata: MetadataStore) -> None:
    svc = _make_service(metadata)
    q = SearchQuery(
        text="anything",
        mode=SearchMode.HYBRID_VISUAL,
        top_k=3,
        visual_weight=1.0,
    )
    results = svc.search(q)
    assert _ids(results) == [("L25_V001", 1), ("L25_V001", 2), ("L25_V001", 3)]


def test_query_validates_visual_weight() -> None:
    with pytest.raises(ValueError):
        SearchQuery(text="x", mode=SearchMode.HYBRID_VISUAL, visual_weight=1.5)