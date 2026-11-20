"""
Tests for Persistent FAISS Vector Store and Hash Verification.
Verifies index creation, disk persistence, reloading,
SHA-256 change detection, and semantic similarity search.
"""

import json
import os
import shutil
import tempfile
import numpy as np
import pytest

from backend.embeddings import EmbeddingGenerator
from backend.ingestion import (
    FAISSVectorStore,
    PDFExtractor,
    RecursiveCharacterTextSplitter,
    compute_file_sha256,
)
from backend.models import DocumentChunk


@pytest.fixture
def temp_index_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_faiss_store_build_save_and_reload(temp_index_dir):
    """Verify building, persisting, and reloading FAISS index and chunk metadata."""
    chunks = [
        DocumentChunk(
            chunk_id="chunk_p1_1",
            source="test_runbook.pdf",
            page_number=1,
            content="Kubernetes Pod eviction policy threshold is 85% memory.",
            char_count=53,
            metadata={"page_number": 1},
        ),
        DocumentChunk(
            chunk_id="chunk_p2_1",
            source="test_runbook.pdf",
            page_number=2,
            content="Severity-1 P0 outage: L1 on-call must ack within 5 minutes.",
            char_count=59,
            metadata={"page_number": 2},
        ),
    ]

    embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    texts = [c.content for c in chunks]
    embeddings = embedder.generate_embeddings(texts, normalize=True)

    # 1. Build and persist
    store = FAISSVectorStore(index_dir=temp_index_dir)
    store.build_index(
        chunks=chunks,
        embeddings=embeddings,
        pdf_path="test_runbook.pdf",
        pdf_sha256="fake_sha256_abcdef",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=900,
        chunk_overlap=120,
        total_pages=2,
    )
    store.save()

    assert os.path.exists(os.path.join(temp_index_dir, "index.faiss"))
    assert os.path.exists(os.path.join(temp_index_dir, "chunks.json"))
    assert os.path.exists(os.path.join(temp_index_dir, "index_manifest.json"))

    # 2. Reload and verify
    reloaded_store = FAISSVectorStore(index_dir=temp_index_dir)
    reloaded_store.load()

    assert reloaded_store.index is not None
    assert reloaded_store.index.ntotal == 2
    assert len(reloaded_store.chunks) == 2
    assert reloaded_store.chunks[0].chunk_id == "chunk_p1_1"
    assert reloaded_store.chunks[1].chunk_id == "chunk_p2_1"

    # 3. Search similarity
    query_vec = embedder.generate_query_embedding("on-call 5 minutes ack", normalize=True)
    results = reloaded_store.search(query_vec, top_k=1)
    assert len(results) == 1
    matched_chunk, score = results[0]
    assert matched_chunk.chunk_id == "chunk_p2_1"
    assert score > 0.4


def test_file_sha256_computation():
    """Verify SHA-256 calculation for a known file."""
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write("TitanCloud SRE Runbook Content")
        f_path = f.name

    try:
        hash_1 = compute_file_sha256(f_path)
        assert len(hash_1) == 64  # SHA-256 hex string is 64 chars

        # Modify file and verify hash changes
        with open(f_path, "a") as f:
            f.write(" - Modified")

        hash_2 = compute_file_sha256(f_path)
        assert hash_1 != hash_2
    finally:
        os.remove(f_path)


def test_is_index_valid_detection(temp_index_dir):
    """Verify is_index_valid detects missing files, hash changes, and config mismatches."""
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write("Original Runbook Content")
        test_pdf = f.name

    try:
        pdf_hash = compute_file_sha256(test_pdf)
        store = FAISSVectorStore(index_dir=temp_index_dir)

        # Before files exist -> invalid
        valid, reason = store.is_index_valid(test_pdf, "all-MiniLM-L6-v2", 900, 120)
        assert not valid
        assert "Index files do not exist" in reason

        # Create dummy index
        dim = 384
        dummy_emb = np.ones((1, dim), dtype=np.float32)
        dummy_chunk = DocumentChunk(
            chunk_id="c1", source="test", page_number=1, content="text", char_count=4
        )
        store.build_index(
            chunks=[dummy_chunk],
            embeddings=dummy_emb,
            pdf_path=test_pdf,
            pdf_sha256=pdf_hash,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=900,
            chunk_overlap=120,
            total_pages=1,
        )
        store.save()

        # Should be valid now
        valid, reason = store.is_index_valid(test_pdf, "all-MiniLM-L6-v2", 900, 120)
        assert valid

        # Config mismatch -> invalid
        valid, reason = store.is_index_valid(test_pdf, "other-model", 900, 120)
        assert not valid

        # PDF modified -> invalid
        with open(test_pdf, "a") as f:
            f.write(" modified")
        valid, reason = store.is_index_valid(test_pdf, "all-MiniLM-L6-v2", 900, 120)
        assert not valid
        assert "PDF SHA-256 hash changed" in reason

    finally:
        os.remove(test_pdf)

