"""
TitanCloud SRE Incident Assistant - Document Retrieval Module
Provides semantic and hybrid retrieval capabilities over the persistent FAISS vector store.
"""

import logging
import os
import re
from typing import List, Optional, Set, Tuple
import numpy as np

from backend.embeddings import EmbeddingGenerator
from backend.ingestion import FAISSVectorStore
from backend.models import ChatMessage, DocumentChunk, SearchResult

logger = logging.getLogger(__name__)


def extract_technical_terms(text: str) -> Set[str]:
    """
    Extracts high-signal technical identifiers, error codes, severity levels,
    and section numbers from text.
    Examples: ERR_PG_REPLICA_LAG, P0, P1, SLA, SLO, Patroni, BGP, Section 9.1, Module 41.0.
    """
    terms = set()
    # Error codes and capitalized identifiers (e.g., ERR_PG_REPLICA_LAG, BGP, OOMKilled)
    code_pattern = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text)
    terms.update(code_pattern)

    # Severity levels (P0, P1, P2, P3, Sev-1, Severity-1)
    sev_pattern = re.findall(r"\b(?:P[0-4]|Sev(?:erity)?-[0-4])\b", text, re.IGNORECASE)
    terms.update([s.upper() for s in sev_pattern])

    # Section / Module references (e.g., Section 9.1, Module 41.0)
    sec_pattern = re.findall(r"\b(?:Section|Module|Clause)\s+\d+(?:\.\d+)?\b", text, re.IGNORECASE)
    terms.update(sec_pattern)

    # Quantitative metrics (e.g., 99.95%, 5 minutes, 15 minutes)
    metric_pattern = re.findall(r"\b\d+(?:\.\d+)?%?\s*(?:minutes?|hours?|seconds?|days?|%)?\b", text, re.IGNORECASE)
    terms.update([m.strip() for m in metric_pattern if len(m.strip()) > 1 and not m.strip().isdigit()])

    return terms


class DocumentRetriever:
    """Retrieves relevant document chunks from the FAISS vector store for a given query."""

    DEFAULT_SCORE_THRESHOLD: float = 0.28

    def __init__(
        self,
        index_dir: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ):
        self.index_dir = (
            index_dir
            or os.getenv("VECTOR_STORE_DIR")
            or "vector_store/faiss_index"
        )
        self.embedding_model_name = (
            embedding_model_name
            or os.getenv("EMBEDDING_MODEL_NAME")
            or EmbeddingGenerator.DEFAULT_MODEL_NAME
        )
        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else float(os.getenv("RELEVANCE_THRESHOLD", self.DEFAULT_SCORE_THRESHOLD))
        )
        self.embedder = EmbeddingGenerator(model_name=self.embedding_model_name)
        self.vector_store = FAISSVectorStore(index_dir=self.index_dir)
        self._is_loaded = False

    def load_index(self) -> bool:
        """Loads the FAISS index and chunk metadata from disk."""
        try:
            self.vector_store.load()
            self._is_loaded = True
            logger.info(f"Retriever loaded index with {len(self.vector_store.chunks)} chunks.")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store from '{self.index_dir}': {e}")
            self._is_loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def resolve_conversational_query(
        self,
        question: str,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> str:
        """
        Resolves follow-up questions by incorporating contextual references
        from recent conversation turns.
        """
        if not conversation_history:
            return question.strip()

        question_clean = question.strip()
        lower_q = question_clean.lower()

        # Follow-up indicators (pronouns, underspecified questions)
        followup_cues = [
            "who", "what about", "how about", "then", "which",
            "this", "that", "it", "they", "those", "these",
            "next", "escalate", "notify", "contact", "timeline",
            "step", "steps", "procedure", "sla", "slo", "lag",
        ]

        is_likely_followup = (
            len(question_clean.split()) <= 6
            or any(re.search(r"\b" + re.escape(cue) + r"\b", lower_q) for cue in followup_cues)
        )

        if not is_likely_followup:
            return question_clean

        # Extract entities / key topics from the most recent user and assistant turns
        context_keywords: List[str] = []
        for msg in reversed(conversation_history[-3:]):
            terms = extract_technical_terms(msg.content)
            for t in terms:
                if t.lower() not in lower_q and t not in context_keywords:
                    context_keywords.append(t)

        if context_keywords:
            prefix = " ".join(context_keywords[:3])
            resolved_query = f"{prefix} {question_clean}"
            logger.info(f"Resolved conversational follow-up query: '{resolved_query}' (original: '{question_clean}')")
            return resolved_query

        return question_clean

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: Optional[float] = None,
        enable_hybrid_boost: bool = True,
    ) -> List[SearchResult]:
        """
        Performs semantic vector search with optional keyword boosting for technical terms.

        Args:
            query: The user search query.
            top_k: Number of most similar chunks to return (default: 4).
            score_threshold: Minimum similarity threshold (overrides instance default if set).
            enable_hybrid_boost: Whether to apply exact keyword matching boost for error codes/terms.

        Returns:
            List of SearchResult objects sorted by similarity score descending.
        """
        if not self._is_loaded:
            loaded = self.load_index()
            if not loaded:
                raise RuntimeError(f"Vector store index not found in '{self.index_dir}'. Please run ingestion.")

        if not query or not query.strip():
            return []

        threshold = score_threshold if score_threshold is not None else self.score_threshold
        query_clean = query.strip()

        # 1. Dense Semantic Search
        # Retrieve extra candidates for hybrid re-ranking
        candidate_count = min(len(self.vector_store.chunks), max(top_k * 3, 10))
        query_vector = self.embedder.generate_query_embedding(query_clean, normalize=True)
        raw_matches = self.vector_store.search(query_vector=query_vector, top_k=candidate_count)

        if not raw_matches:
            return []

        # 2. Hybrid Keyword Boosting (for exact error codes, service names, sections)
        tech_terms = extract_technical_terms(query_clean) if enable_hybrid_boost else set()

        scored_results: List[Tuple[DocumentChunk, float]] = []
        for chunk, semantic_score in raw_matches:
            final_score = float(semantic_score)

            if tech_terms:
                chunk_upper = chunk.content.upper()
                term_matches = sum(1 for term in tech_terms if term.upper() in chunk_upper)
                if term_matches > 0:
                    # Apply proportional boost for exact technical matches (capped at 0.15)
                    boost = min(0.15, 0.05 * term_matches)
                    final_score = min(1.0, final_score + boost)

            if final_score >= threshold:
                scored_results.append((chunk, final_score))

        # Sort descending by final score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Slice to top_k
        top_results = scored_results[:top_k]

        results: List[SearchResult] = [
            SearchResult(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                page_number=chunk.page_number,
                content=chunk.content,
                score=round(score, 4),
                metadata=chunk.metadata,
            )
            for chunk, score in top_results
        ]

        logger.info(
            f"Retrieved {len(results)} chunks for query '{query_clean[:40]}...' (threshold={threshold})"
        )
        return results
