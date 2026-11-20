"""
Tests for Streamlit Frontend Architecture and Compliance.
Verifies helper functions, backend integration, error handling,
and compliance with strict restrictions (NO file uploader, NO document management).
"""

from unittest.mock import patch, MagicMock
import pytest
from frontend.app import (
    BACKEND_URL,
    clear_session_history,
    get_backend_health,
    get_backend_stats,
    send_chat_message,
)


def test_no_forbidden_upload_or_document_controls():
    """Verify frontend code contains NO file_uploader or document management."""
    with open("frontend/app.py", "r", encoding="utf-8") as f:
        code = f.read()

    assert "file_uploader" not in code, "Violation: file_uploader found in frontend/app.py"
    assert "upload_file" not in code, "Violation: upload_file found in frontend/app.py"
    assert "rebuild_index" not in code, "Violation: rebuild_index found in frontend/app.py"
    assert "ingest_pdf" not in code, "Violation: ingest_pdf found in frontend/app.py"


@patch("requests.get")
def test_get_backend_health_success(mock_get):
    """Verify get_backend_health returns data when backend responds 200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "healthy",
        "service": "titancloud-sre-incident-assistant",
        "vector_store_loaded": True,
    }
    mock_get.return_value = mock_resp

    data = get_backend_health()
    assert data is not None
    assert data["status"] == "healthy"
    assert data["vector_store_loaded"] is True


@patch("requests.get")
def test_get_backend_health_failure(mock_get):
    """Verify get_backend_health returns None when backend is unreachable."""
    mock_get.side_effect = Exception("Connection refused")
    data = get_backend_health()
    assert data is None


@patch("requests.get")
def test_get_backend_stats_success(mock_get):
    """Verify get_backend_stats returns statistics when backend responds 200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "indexed_document": "titancloud_sre_incident_runbook_50p.pdf",
        "total_pages": 50,
        "total_chunks": 100,
    }
    mock_get.return_value = mock_resp

    data = get_backend_stats()
    assert data is not None
    assert data["total_pages"] == 50
    assert data["total_chunks"] == 100


@patch("requests.post")
def test_send_chat_message_success(mock_post):
    """Verify send_chat_message forwards payload and returns structured response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answer": "Acknowledged within 5 minutes under Section 9.1 [Page 41].",
        "sources": [
            {
                "filename": "titancloud_sre_incident_runbook_50p.pdf",
                "page_number": 41,
                "text": "When a Severity-1 (P0) alert triggers...",
            }
        ],
        "retrieval_status": "success",
        "session_id": "test-session",
        "confidence": 0.72,
        "is_sufficient": True,
    }
    mock_post.return_value = mock_resp

    result = send_chat_message(
        question="What is the P0 escalation timeline?",
        session_id="test-session",
        chat_history=[],
    )

    assert result["retrieval_status"] == "success"
    assert "Page 41" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["page_number"] == 41


@patch("requests.post")
def test_send_chat_message_connection_error(mock_post):
    """Verify send_chat_message handles backend connection error cleanly."""
    import requests
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

    result = send_chat_message(
        question="What is the SLA?",
        session_id="test-session",
        chat_history=[],
    )

    assert result["retrieval_status"] == "error"
    assert "Cannot connect to the backend server" in result["answer"]

