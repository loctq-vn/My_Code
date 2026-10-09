"""Offline ingestion layer for the Video Retrieval UI.

Builds the LanceDB search index from caption JSONs + pre-computed bge-m3
embeddings (:mod:`ingestion.embedder`), and provides the query-side encoder.
"""

from .embedder import Embedder

__all__ = ["Embedder"]