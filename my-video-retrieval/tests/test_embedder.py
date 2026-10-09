"""Light tests for :class:`ingestion.embedder.Embedder`.

These deliberately avoid loading the SentenceTransformer (heavy): they only
exercise the on-disk ``.npy`` path of ``load_shot_embeddings``. The real-model
query-encoding gate is run separately via the ingestion smoke script.

Run with:  python -m pytest tests -v
"""

import numpy as np
import pytest

from ingestion.embedder import Embedder


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def test_load_shot_embeddings_shape(embedder: Embedder) -> None:
    arr = embedder.load_shot_embeddings("L25_V001", expected=19)
    assert arr.shape == (19, 1024)
    assert arr.dtype == np.float32


def test_load_shot_embeddings_normalized(embedder: Embedder) -> None:
    arr = embedder.load_shot_embeddings("L25_V001", expected=19)
    norms = np.linalg.norm(arr, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_load_shot_embeddings_unknown_video(embedder: Embedder) -> None:
    with pytest.raises(ValueError):
        embedder.load_shot_embeddings("DOES_NOT_EXIST", expected=0)