"""
Tests for PDF Extraction module.
Verifies PDF readability, exact page count, text extraction quality,
page metadata preservation, and error handling.
"""

import os
import pytest
from backend.ingestion import PDFExtractor
from backend.models import DocumentPage

PDF_PATH = "data/titancloud_sre_incident_runbook_50p.pdf"


def test_pdf_file_exists():
    """Verify that the official runbook PDF exists in data/ directory."""
    assert os.path.exists(PDF_PATH), f"PDF file does not exist at {PDF_PATH}"
    assert os.path.getsize(PDF_PATH) > 0, "PDF file is empty"


def test_pdf_page_count_and_extraction():
    """Verify that the PDF has exactly 50 pages and all are extracted."""
    extractor = PDFExtractor(PDF_PATH)
    pages = extractor.extract_pages()

    assert len(pages) == 50, f"Expected 50 pages, but got {len(pages)}"
    # Verify 1-based page numbering
    for i, page in enumerate(pages):
        assert page.page_number == i + 1, f"Expected page {i+1}, got {page.page_number}"
        assert page.source == "titancloud_sre_incident_runbook_50p.pdf"
        assert page.char_count > 0, f"Page {page.page_number} unexpectedly empty"


def test_pdf_extraction_content_quality():
    """Verify that key domain terms, headers, and codes are preserved."""
    extractor = PDFExtractor(PDF_PATH)
    pages = extractor.extract_pages()

    page_1_text = pages[0].text
    assert "TitanCloud Enterprise SRE & Incident Response Runbook" in page_1_text
    assert "CHAPTER 1" in page_1_text
    assert "Page 1 of 50" in page_1_text

    # Page 41 has the Severity-1 escalation procedures
    page_41_text = pages[40].text
    assert "SEVERITY-1 (P0) ON-CALL ACKNOWLEDGEMENT" in page_41_text
    assert "PagerDuty" in page_41_text
    assert "5 minutes" in page_41_text


def test_clean_text_preserves_technical_terms():
    """Verify clean_text removes excess whitespace but retains structure and terms."""
    raw = "  TitanCloud   SRE   Runbook\n\n\n\nERR_PG_REPLICA_LAG  =  99.95%   \n  "
    cleaned = PDFExtractor.clean_text(raw)
    assert "TitanCloud   SRE   Runbook" in cleaned
    assert "ERR_PG_REPLICA_LAG  =  99.95%" in cleaned
    assert "\n\n\n" not in cleaned


def test_extractor_missing_file_raises_error():
    """Verify that initializing with non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        PDFExtractor("data/non_existent_runbook.pdf")

