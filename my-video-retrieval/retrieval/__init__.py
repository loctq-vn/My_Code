"""Online retrieval layer for the Video Retrieval UI.

Defines the retriever contract (:mod:`retrieval.base`) and concrete
implementations that search the LanceDB index (vector / BM25 / hybrid-RRF)
and are consumed by :mod:`retrieval.search_service`.
"""

from .base import BaseRetriever, RawHit, SearchMode
from .lancedb_retriever import LanceDBRetriever
from .search_service import SearchService

__all__ = ["BaseRetriever", "RawHit", "SearchMode", "LanceDBRetriever", "SearchService"]