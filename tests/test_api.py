"""
Tests for FastAPI Backend Application.
Verifies /health, /chat, /chat/clear, /stats, /documents, and /search endpoints,
including session management, error handling, and request validation.
"""

from fastapi.testclient import TestClient
import pytest

from backend.main import app, session_conversations

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns operational status of all components."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "titancloud-sre-incident-assistant"
    assert data["version"] == "1.0.0"
    assert data["vector_store_loaded"] is True
    assert data["total_chunks"] == 100
    assert data["embedding_model"] == "all-MiniLM-L6-v2"
    assert data["embedding_dimension"] == 384
    assert "gemini_configured" in data


def test_stats_endpoint():
    """Verify GET /stats returns actual corpus and vector store statistics."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()

    assert data["indexed_document"] == "titancloud_sre_incident_runbook_50p.pdf"
    assert data["total_pages"] == 50
    assert data["total_chunks"] == 100
    assert data["chunk_size"] == 900
    assert data["chunk_overlap"] == 120
    assert data["avg_chunk_length"] > 700
    assert data["embedding_model"] == "all-MiniLM-L6-v2"
    assert data["embedding_dimension"] == 384
    assert data["vector_database_status"] == "active"
    assert len(data["pdf_sha256"]) == 64


def test_documents_endpoint():
    """Verify GET /documents returns metadata for the single fixed runbook."""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()

    assert data["filename"] == "titancloud_sre_incident_runbook_50p.pdf"
    assert data["total_pages"] == 50
    assert data["total_chunks"] == 100
    assert data["file_size_bytes"] > 50000
    assert len(data["pdf_sha256"]) == 64
    assert data["is_read_only"] is True


def test_chat_endpoint_valid_question():
    """Verify POST /chat answers questions and returns citations with page numbers."""
    payload = {
        "question": "What is the escalation timeline when a Severity-1 (P0) alert triggers in PagerDuty?",
        "session_id": "test-session-123",
        "top_k": 3,
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert data["session_id"] == "test-session-123"
    assert data["retrieval_status"] == "success"
    assert len(data["sources"]) > 0
    # Page 41 must be cited for Severity-1 timeline
    assert any(s["page_number"] == 41 for s in data["sources"])
    assert data["is_sufficient"] is True


def test_chat_endpoint_multi_turn_followup():
    """Verify multi-turn conversational follow-up via session_id."""
    session_id = "test-session-followup"

    # Turn 1
    resp1 = client.post(
        "/chat",
        json={"question": "What is the procedure for a Severity-1 (P0) incident?", "session_id": session_id},
    )
    assert resp1.status_code == 200

    # Turn 2: Follow-up question referencing the prior turn
    resp2 = client.post(
        "/chat",
        json={"question": "Who needs to be notified if unacknowledged after 15 minutes?", "session_id": session_id},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()

    assert len(data2["sources"]) > 0
    # Must retrieve Page 41 (escalation to VP of Infrastructure)
    assert any(s["page_number"] == 41 for s in data2["sources"])


def test_chat_clear_session():
    """Verify POST /chat/clear removes session context."""
    session_id = "session-to-clear"
    client.post("/chat", json={"question": "What is the SLA?", "session_id": session_id})
    assert session_id in session_conversations

    clear_resp = client.post("/chat/clear", json={"session_id": session_id})
    assert clear_resp.status_code == 200
    assert clear_resp.json()["cleared"] is True
    assert session_id not in session_conversations

    # Second clear on same session returns cleared=False
    clear_again = client.post("/chat/clear", json={"session_id": session_id})
    assert clear_again.status_code == 200
    assert clear_again.json()["cleared"] is False


def test_chat_insufficient_context():
    """Verify out-of-domain questions return standard insufficient context message."""
    payload = {
        "question": "What is the capital city of France?",
        "session_id": "out-of-domain-session",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["answer"] == "The uploaded SRE runbook does not contain sufficient information to answer this question."
    assert data["retrieval_status"] == "no_context"
    assert data["is_sufficient"] is False
    assert len(data["sources"]) == 0


def test_chat_validation_errors():
    """Verify invalid payloads return HTTP 422."""
    # Empty question
    resp_empty = client.post("/chat", json={"question": ""})
    assert resp_empty.status_code == 422

    # Missing question field
    resp_missing = client.post("/chat", json={})
    assert resp_missing.status_code == 422


def test_search_endpoint():
    """Verify GET /search developer endpoint."""
    resp = client.get("/search?q=ERR_PG_REPLICA_LAG&top_k=2")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) > 0
    assert results[0]["page_number"] == 22
    assert "ERR_PG_REPLICA_LAG" in results[0]["content"]

    # Missing query parameter
    resp_bad = client.get("/search")
    assert resp_bad.status_code == 422

