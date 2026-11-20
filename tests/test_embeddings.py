"""
Tests for Embedding Generation module.
Verifies all-MiniLM-L6-v2 model loading, embedding dimensions,
normalization, batch generation, and query embedding.
"""

import numpy as np
import pytest
from backend.embeddings import EmbeddingGenerator


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingGenerator(model_name="all-MiniLM-L6-v2")


def test_model_loading_and_dimension(embedder):
    """Verify model loads with dimension 384."""
    assert embedder.dimension == 384
    assert embedder.model_name == "all-MiniLM-L6-v2"


def test_embedding_generation_shape_and_dtype(embedder):
    """Verify embedding output shape, dtype float32, and normalization."""
    texts = [
        "Kubernetes pod eviction and OOMKilled troubleshooting.",
        "PostgreSQL Patroni high-availability cluster failover.",
        "Severity-1 alert escalation timeline: 5 minutes to acknowledge.",
    ]
    embeddings = embedder.generate_embeddings(texts, batch_size=2, normalize=True)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 384)
    assert embeddings.dtype == np.float32

    # Check L2 unit norm
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_query_embedding(embedder):
    """Verify single query embedding generation."""
    query = "How to handle ERR_PG_REPLICA_LAG?"
    query_emb = embedder.generate_query_embedding(query, normalize=True)

    assert query_emb.shape == (1, 384)
    assert query_emb.dtype == np.float32
    norm = np.linalg.norm(query_emb[0])
    np.testing.assert_allclose(norm, 1.0, atol=1e-5)


def test_empty_query_raises_error(embedder):
    """Verify empty query string raises ValueError."""
    with pytest.raises(ValueError):
        embedder.generate_query_embedding("")
    with pytest.raises(ValueError):
        embedder.generate_query_embedding("   ")


def test_empty_texts_list(embedder):
    """Verify empty list returns empty array of correct shape."""
    empty_embs = embedder.generate_embeddings([])
    assert empty_embs.shape == (0, 384)

