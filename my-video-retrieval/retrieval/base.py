"""Retriever contract for the Video Retrieval UI.

A retriever turns a :class:`core.schemas.SearchQuery` into raw ranked hits.
It deliberately works on *ids + scores*, not on full :class:`core.schemas.ShotChunk`
objects — enrichment (titles, local paths, thumbnails) is the job of
:mod:`retrieval.search_service` via :class:`core.metadata_store.MetadataStore`,
so the same retrievers stay swappable (LanceDB today, FAISS later) without
touching the UI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from core.schemas import SearchMode, SearchQuery


@dataclass(frozen=True)
class RawHit:
    """A raw retrieval hit from any backend: stable row id + score."""

    document_id: str  # row id, e.g. "L25_V001:3"
    score: float  # higher = more relevant (already normalized/positive)


class BaseRetriever(ABC):
    """Abstract interface every search backend must implement.

    Implementations only need to resolve a ``SearchQuery`` to ``RawHit``s.
    """

    #: Search modes this backend actually supports.
    supported_modes: Tuple[SearchMode, ...] = (SearchMode.VECTOR, SearchMode.KEYWORD, SearchMode.HYBRID)

    @abstractmethod
    def search(self, query: SearchQuery) -> List[RawHit]:
        """Ranked hits for ``query`` (best first, score >= 0, most-to-least relevant).

        ``query.top_k`` is the maximum number of hits to return.
        """

    def supports(self, mode: SearchMode) -> bool:
        return mode in self.supported_modes