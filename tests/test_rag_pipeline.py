"""
Tests for RAG Pipeline, Grounded Prompting, Gemini Integration, and Citations.
Verifies citation accuracy, insufficient context handling, conversational follow-up,
and mocked LLM generation.
"""

from unittest.mock import MagicMock, patch
import pytest

from backend.models import ChatMessage, RAGRequest, RAGResponse, SearchResult
from backend.rag_pipeline import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    SYSTEM_INSTRUCTION,
    GeminiGenerator,
    RAGPipeline,
)
from backend.retrieval import DocumentRetriever


@pytest.fixture
def mock_retriever():
    retriever = MagicMock(spec=DocumentRetriever)
    retriever.is_loaded = True
    retriever.resolve_conversational_query = lambda question, conversation_history=None: question
    return retriever


@pytest.fixture
def mock_generator():
    gen = MagicMock(spec=GeminiGenerator)
    gen.is_available = True
    return gen


def test_system_prompt_rules_presence():
    """Verify SYSTEM_INSTRUCTION includes all 12 enterprise rules."""
    assert "TitanCloud SRE Incident Assistant" in SYSTEM_INSTRUCTION
    assert "RULES:" in SYSTEM_INSTRUCTION
    assert "1. Answer using the retrieved document context." in SYSTEM_INSTRUCTION
    assert "2. Never invent information." in SYSTEM_INSTRUCTION
    assert "3. Never fabricate SLA values, error codes" in SYSTEM_INSTRUCTION
    assert "4. If the retrieved context is insufficient" in SYSTEM_INSTRUCTION
    assert "5. Preserve exact technical values" in SYSTEM_INSTRUCTION
    assert "7. Cite the original PDF page numbers." in SYSTEM_INSTRUCTION
    assert "10. Never claim to have executed an incident response action." in SYSTEM_INSTRUCTION
    assert "11. Do not use general LLM knowledge" in SYSTEM_INSTRUCTION
    assert "12. Do not invent facts." in SYSTEM_INSTRUCTION


def test_insufficient_context_when_retrieval_empty(mock_retriever, mock_generator):
    """Verify that when no chunks meet relevance threshold, standard insufficient message is returned."""
    mock_retriever.retrieve.return_value = []

    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    response = pipeline.answer("What is the recipe for chocolate cake?")

    assert response.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert response.sources == []
    assert not response.is_sufficient
    assert response.confidence == 0.0
    # Gemini should not even be called when retrieval is empty
    mock_generator.generate.assert_not_called()


def test_citation_accuracy_and_page_number_provenance(mock_retriever, mock_generator):
    """Verify citations originate strictly from retrieved chunk metadata."""
    retrieved = [
        SearchResult(
            chunk_id="chunk_p41_1",
            source="titancloud_sre_incident_runbook_50p.pdf",
            page_number=41,
            content="When a Severity-1 (P0) production outage alert triggers in PagerDuty, L1 has 5 minutes.",
            score=0.72,
        ),
        SearchResult(
            chunk_id="chunk_p9_1",
            source="titancloud_sre_incident_runbook_50p.pdf",
            page_number=9,
            content="TitanCloud guarantees a 99.95% monthly uptime Service Level Agreement.",
            score=0.65,
        ),
    ]
    mock_retriever.retrieve.return_value = retrieved
    mock_generator.generate.return_value = (
        "According to Page 41, the L1 engineer must acknowledge a P0 alert within 5 minutes. "
        "Separately, Page 9 documents a 99.95% monthly uptime SLA."
    )

    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    response = pipeline.answer("What are the P0 escalation time and uptime SLA?")

    assert response.is_sufficient
    assert len(response.sources) == 2
    assert response.sources[0].page_number == 41
    assert response.sources[0].filename == "titancloud_sre_incident_runbook_50p.pdf"
    assert "5 minutes" in response.sources[0].text

    assert response.sources[1].page_number == 9
    assert response.sources[1].filename == "titancloud_sre_incident_runbook_50p.pdf"
    assert "99.95%" in response.sources[1].text


def test_missing_api_key_graceful_handling(mock_retriever):
    """Verify missing Gemini API key does not crash and provides retrieved citations."""
    retrieved = [
        SearchResult(
            chunk_id="chunk_p22_1",
            source="titancloud_sre_incident_runbook_50p.pdf",
            page_number=22,
            content="Runbook for ERR_PG_REPLICA_LAG: verify replication stream.",
            score=0.68,
        )
    ]
    mock_retriever.retrieve.return_value = retrieved

    # Generator without valid key
    unauth_gen = GeminiGenerator(api_key="")
    pipeline = RAGPipeline(retriever=mock_retriever, generator=unauth_gen)
    response = pipeline.answer("How do I fix ERR_PG_REPLICA_LAG?")

    assert "GEMINI_API_KEY is not configured" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0].page_number == 22


def test_llm_reports_insufficient_context(mock_retriever, mock_generator):
    """Verify pipeline detects when LLM states the runbook does not contain the answer."""
    retrieved = [
        SearchResult(
            chunk_id="chunk_p1_1",
            source="titancloud_sre_incident_runbook_50p.pdf",
            page_number=1,
            content="Kubernetes control plane telemetry documentation.",
            score=0.45,
        )
    ]
    mock_retriever.retrieve.return_value = retrieved
    mock_generator.generate.return_value = (
        "The uploaded SRE runbook does not contain sufficient information to answer this question."
    )

    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    response = pipeline.answer("What is the personal phone number of the SRE Director?")

    assert response.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert not response.is_sufficient
    assert response.sources == []


def test_conversational_followup_invocation():
    """Verify end-to-end follow-up handling with real retriever and mock generator."""
    real_retriever = DocumentRetriever()
    real_retriever.load_index()

    mock_gen = MagicMock(spec=GeminiGenerator)
    mock_gen.is_available = True
    mock_gen.generate.return_value = "Under Section 9.1 on Page 41, the alert escalates to L2 Staff SRE Lead at 5 minutes."

    pipeline = RAGPipeline(retriever=real_retriever, generator=mock_gen)

    history = [
        ChatMessage(role="user", content="What is the escalation timeline for a Severity-1 (P0) incident?"),
        ChatMessage(role="assistant", content="Under Section 9.1 on Page 41, L1 has 5 minutes to acknowledge."),
    ]
    followup_q = "Who is paged next if unacknowledged?"

    response = pipeline.answer(question=followup_q, conversation_history=history, top_k=2)

    assert response.is_sufficient
    assert len(response.sources) > 0
    # Top retrieved page should be Page 41
    assert any(s.page_number == 41 for s in response.sources)
