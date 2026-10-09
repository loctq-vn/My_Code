"""Tests for the video / dataset multi-filters added for search-scope limiting.

Covers:
* ``SearchQuery`` normalisation of ``video_ids`` / ``datasets``,
* ``LanceDBRetriever._apply_filters`` emitting ``IN (...)`` where clauses,
* ``KeyframeVectorStore.iter_videos`` / ``search_shots`` honouring the set
  of selected videos.

Run with:  python -m pytest tests -v
"""

import pytest

from config.settings import SETTINGS
from core.metadata_store import MetadataStore
from core.schemas import SearchMode, SearchQuery
from retrieval.lancedb_retriever import LanceDBRetriever
from retrieval.visual_scorer import KeyframeVectorStore


@pytest.fixture(scope="module")
def metadata() -> MetadataStore:
    return MetadataStore()


# -------------------------------------------------------- SearchQuery --------
def test_shortlists_empty_collection_normalized_to_none() -> None:
    q = SearchQuery(text="x", video_ids=[], datasets=[])
    assert q.video_ids is None
    assert q.datasets is None


def test_shortlists_tuple_ized_and_deduped() -> None:
    q = SearchQuery(text="x", video_ids=("A", "B", "A"), datasets=["L25_data", "L25_data"])
    assert q.video_ids == ("A", "B")
    assert q.datasets == ("L25_data",)


# ------------------------------------------------------------ filters -----
def _retriever(metadata: MetadataStore) -> LanceDBRetriever:
    return LanceDBRetriever(settings=SETTINGS, metadata=metadata)


def test_apply_filters_no_filters(metadata: MetadataStore) -> None:
    r = _retriever(metadata)
    assert r._apply_filters(SearchQuery(text="x")) is None


def test_apply_filters_video_ids_in_clause(metadata: MetadataStore) -> None:
    r = _retriever(metadata)
    q = SearchQuery(text="x", video_ids=("L30_V007", "L25_V001"))
    where = r._apply_filters(q)
    assert where == 'video_id IN ("L25_V001", "L30_V007")'


def test_apply_filters_dataset_and_video_merge(metadata: MetadataStore) -> None:
    r = _retriever(metadata)
    # singular + plural should merge into a single IN clause each.
    q = SearchQuery(
        text="x",
        dataset="L25_data",
        datasets=(),
        video_id="L25_V001",
        video_ids=("L25_V001", "L25_V002"),
    )
    where = r._apply_filters(q)
    assert 'dataset IN ("L25_data")' in where
    assert 'video_id IN ("L25_V001", "L25_V002")' in where


def test_apply_filters_times_still_work(metadata: MetadataStore) -> None:
    r = _retriever(metadata)
    q = SearchQuery(text="x", video_ids=("L25_V001",), start_time=10.0, end_time=20.0)
    where = r._apply_filters(q)
    assert where == 'video_id IN ("L25_V001") AND start_time >= 10.0 AND end_time <= 20.0'


# ------------------------------------------------------------- visual -----
def test_iter_videos_respects_video_ids(metadata: MetadataStore) -> None:
    store = KeyframeVectorStore(settings=SETTINGS, metadata=metadata)
    got = store.iter_videos(video_ids=["L25_V001", "L30_V007"])
    assert set(got) == {"L25_V001", "L30_V007"}
    assert store.iter_videos(video_id="L25_V001", video_ids=["L30_V007"]) == ["L25_V001"]


def test_iter_videos_intersects_dataset_and_video_scope(metadata: MetadataStore) -> None:
    store = KeyframeVectorStore(settings=SETTINGS, metadata=metadata)
    # Choose a video that does NOT belong to L25_data -> intersection is empty.
    assert store.iter_videos(datasets=["L25_data"], video_ids=["L30_V007"]) == []
    # A real L25 video survives the intersection.
    assert store.iter_videos(datasets=["L25_data"], video_ids=["L25_V001"]) == ["L25_V001"]
    # Multi-dataset scope unions its members (all known videos).
    assert set(store.iter_videos(datasets=["L25_data", "L30_data"])) == set(
        metadata.all_video_ids()
    )


def test_search_shots_restricted_to_video_ids(metadata: MetadataStore) -> None:
    store = KeyframeVectorStore(settings=SETTINGS, metadata=metadata)
    vec = store.vector_for("L25_V001", 4)
    results = store.search_shots(vec, video_ids=["L25_V001"], top_k=5)
    assert results
    assert {vid for _, vid, _ in results} == {"L25_V001"}


# ----------------------------------------------------- new search modes ----
def test_new_modes_supported() -> None:
    from core.schemas import SearchMode as SM
    from retrieval.lancedb_retriever import LanceDBRetriever as RR

    assert SM.CAPTION_EMBEDDING in RR.supported_modes
    assert SM.SUBTITLE_RAW in RR.supported_modes


def test_subtitle_raw_search_matches_vietnamese(metadata: MetadataStore) -> None:
    from core.schemas import SearchMode, SearchQuery
    from retrieval.lancedb_retriever import LanceDBRetriever

    r = LanceDBRetriever(metadata=metadata)
    q = SearchQuery(text="duc anh", mode=SearchMode.SUBTITLE_RAW, top_k=5)
    hits = r.search(q)
    assert hits and all(h.score > 0 for h in hits)


def test_subtitle_raw_restricted_to_video_ids(metadata: MetadataStore) -> None:
    from core.schemas import SearchMode, SearchQuery
    from retrieval.lancedb_retriever import LanceDBRetriever

    r = LanceDBRetriever(metadata=metadata)
    q = SearchQuery(
        text="duc anh", mode=SearchMode.SUBTITLE_RAW, top_k=5, video_ids=("L30_V060",)
    )
    hits = r.search(q)
    assert hits and all(h.document_id.startswith("L30_V060:") for h in hits)


def test_subtitle_raw_no_scorable_terms_no_hits(metadata: MetadataStore) -> None:
    from core.schemas import SearchMode, SearchQuery
    from retrieval.lancedb_retriever import LanceDBRetriever

    r = LanceDBRetriever(metadata=metadata)
    # Single-char tokens are all dropped -> no scorable terms -> no hits.
    hits = r._subtitle_search(SearchQuery(text="x z", mode=SearchMode.SUBTITLE_RAW))
    assert hits == []