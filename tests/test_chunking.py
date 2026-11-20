"""
Tests for Text Chunking module.
Verifies recursive character splitting, chunk sizes, overlap,
metadata preservation, and boundary guarantees.
"""

import pytest
from backend.ingestion import RecursiveCharacterTextSplitter
from backend.models import DocumentPage


def test_chunking_default_configuration():
    """Verify splitter initializes with default 900 size and 120 overlap."""
    splitter = RecursiveCharacterTextSplitter()
    assert splitter.chunk_size == 900
    assert splitter.chunk_overlap == 120


def test_chunking_invalid_configuration():
    """Verify invalid configurations are rejected."""
    with pytest.raises(ValueError):
        RecursiveCharacterTextSplitter(chunk_size=0)
    with pytest.raises(ValueError):
        RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    with pytest.raises(ValueError):
        RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=-10)


def test_chunk_generation_and_metadata_preservation():
    """Verify chunk generation preserves source, 1-based page_number, and chunk_id."""
    sample_page = DocumentPage(
        page_number=41,
        text=(
            "TitanCloud SRE Runbook Section 9.1: Severity-1 Escalation.\n"
            "When a Severity-1 (P0) production outage alert triggers in PagerDuty:\n"
            "1. L1 On-Call has 5 minutes to acknowledge.\n"
            "2. If unacknowledged, PagerDuty pages L2 Staff SRE Lead.\n"
            "3. If unacknowledged for 15 minutes, escalates to VP of Infrastructure."
        ),
        source="titancloud_sre_incident_runbook_50p.pdf",
        char_count=280,
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    chunks = splitter.split_pages([sample_page])

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.source == "titancloud_sre_incident_runbook_50p.pdf"
        assert chunk.page_number == 41
        assert chunk.chunk_id.startswith("chunk_p41_")
        assert len(chunk.content) <= 200
        assert chunk.metadata["page_number"] == 41
        assert chunk.metadata["source"] == "titancloud_sre_incident_runbook_50p.pdf"


def test_no_cross_page_chunk_spill():
    """Verify that chunks strictly retain their own page number and do not bleed into others."""
    page_1 = DocumentPage(page_number=1, text="Content of Page 1.", source="runbook.pdf", char_count=18)
    page_2 = DocumentPage(page_number=2, text="Content of Page 2.", source="runbook.pdf", char_count=18)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_pages([page_1, page_2])

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_id == "chunk_p1_1"
    assert "Content of Page 1" in chunks[0].content

    assert chunks[1].page_number == 2
    assert chunks[1].chunk_id == "chunk_p2_1"
    assert "Content of Page 2" in chunks[1].content

