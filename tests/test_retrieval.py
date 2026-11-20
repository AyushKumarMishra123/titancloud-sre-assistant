"""
Tests for Document Retrieval and Hybrid Search.
Verifies semantic retrieval, keyword boosting for error codes,
metadata preservation, configurable top-k, and follow-up query rewriting.
"""

import pytest
from backend.models import ChatMessage, SearchResult
from backend.retrieval import DocumentRetriever, extract_technical_terms


@pytest.fixture(scope="module")
def retriever():
    r = DocumentRetriever()
    r.load_index()
    return r


def test_retriever_is_loaded(retriever):
    """Verify retriever successfully loads FAISS index."""
    assert retriever.is_loaded
    assert len(retriever.vector_store.chunks) == 100


def test_retrieval_p0_escalation_timeline(retriever):
    """Verify P0 escalation query retrieves Page 41 as top result."""
    query = "What is the escalation timeline when a Severity-1 (P0) alert triggers in PagerDuty?"
    results = retriever.retrieve(query, top_k=3)

    assert len(results) > 0
    top = results[0]
    assert top.page_number == 41
    assert "SEVERITY-1 (P0)" in top.content
    assert "5 minutes" in top.content
    assert top.score > 0.5


def test_retrieval_enterprise_uptime_sla(retriever):
    """Verify SLA query retrieves Page 9 containing 99.95% guarantee."""
    query = "What uptime guarantee SLA does TitanCloud provide to enterprise customers?"
    results = retriever.retrieve(query, top_k=3)

    assert len(results) > 0
    top = results[0]
    assert top.page_number == 9
    assert "99.95%" in top.content
    assert top.score > 0.5


def test_retrieval_error_code_boosting(retriever):
    """Verify exact error code ERR_PG_REPLICA_LAG retrieves Page 22."""
    query = "How to resolve ERR_PG_REPLICA_LAG in postgresql clusters?"
    results = retriever.retrieve(query, top_k=3, enable_hybrid_boost=True)

    assert len(results) > 0
    top = results[0]
    assert top.page_number == 22
    assert "ERR_PG_REPLICA_LAG" in top.content


def test_retrieval_out_of_domain_filtered(retriever):
    """Verify completely out-of-domain query returns empty list above threshold."""
    query = "Who won the FIFA World Cup in 2022?"
    results = retriever.retrieve(query, top_k=4, score_threshold=0.30)
    assert len(results) == 0


def test_configurable_top_k(retriever):
    """Verify top_k parameter limits results accurately."""
    query = "Kubernetes control plane telemetry"
    results_2 = retriever.retrieve(query, top_k=2)
    assert len(results_2) == 2

    results_4 = retriever.retrieve(query, top_k=4)
    assert len(results_4) == 4


def test_extract_technical_terms():
    """Verify regex term extractor captures codes, sevs, and section references."""
    sample = "Under Section 9.1, when ERR_PG_REPLICA_LAG or a P0 outage occurs within 5 minutes"
    terms = extract_technical_terms(sample)
    assert "ERR_PG_REPLICA_LAG" in terms
    assert "P0" in terms
    assert "Section 9.1" in terms


def test_conversational_followup_resolution(retriever):
    """Verify follow-up query rewriting incorporates context from prior messages."""
    history = [
        ChatMessage(role="user", content="What is the procedure for a Severity-1 P0 incident?"),
        ChatMessage(role="assistant", content="Under Section 9.1 on Page 41, L1 engineer has 5 minutes to acknowledge."),
    ]
    followup = "Who needs to be notified if unacknowledged after 15 minutes?"
    resolved = retriever.resolve_conversational_query(followup, history)

    assert "P0" in resolved or "SEVERITY-1" in resolved or "Page 41" in resolved
    assert "Who needs to be notified" in resolved

