"""
TitanCloud SRE Incident Assistant - RAG Pipeline Module
Orchestrates semantic retrieval, conversational follow-up resolution,
grounded prompt construction, Google Gemini API synthesis, and programmatic citations.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from backend.models import ChatMessage, RAGRequest, RAGResponse, SearchResult, SourceCitation
from backend.retrieval import DocumentRetriever

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION: str = (
    "You are TitanCloud SRE Incident Assistant, an enterprise Site Reliability Engineering knowledge assistant.\n\n"
    "Your authoritative knowledge source is the retrieved context from the TitanCloud SRE Incident Runbook.\n\n"
    "RULES:\n\n"
    "1. Answer using the retrieved document context.\n"
    "2. Never invent information.\n"
    "3. Never fabricate SLA values, error codes, escalation contacts, or troubleshooting steps.\n"
    "4. If the retrieved context is insufficient, clearly state that the runbook does not contain sufficient information.\n"
    "5. Preserve exact technical values, thresholds, timings, and severity levels.\n"
    "6. Explain documented troubleshooting procedures step by step.\n"
    "7. Cite the original PDF page numbers.\n"
    "8. Combine relevant information from multiple retrieved sections when necessary.\n"
    "9. Distinguish documented procedures from general recommendations.\n"
    "10. Never claim to have executed an incident response action.\n"
    "11. Do not use general LLM knowledge to fill gaps in the document.\n"
    "12. Do not invent facts."
)

INSUFFICIENT_CONTEXT_MESSAGE: str = (
    "The uploaded SRE runbook does not contain sufficient information to answer this question."
)


class GeminiGenerator:
    """Interface to Google Gemini API using the official Google GenAI SDK."""

    DEFAULT_MODEL: str = "gemini-3.8-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GEMINI_API_KEY")

        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or self.DEFAULT_MODEL
        )
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Google GenAI SDK client if API key is valid."""
        if not self.api_key or self.api_key.strip() in ("", "your_api_key", "your_gemini_api_key_here"):
            logger.warning("Gemini API key is not configured or is a placeholder.")
            self._client = None
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key.strip())
            logger.info(f"Initialized Google GenAI client with model '{self.model_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI client: {e}")
            self._client = None

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def generate(
        self,
        prompt: str,
        system_instruction: str = SYSTEM_INSTRUCTION,
        temperature: float = 0.0,
    ) -> str:
        """
        Calls Gemini to generate a response for the given prompt.
        """
        if not self.is_available:
            raise RuntimeError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."
            )

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
            )

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            if response and response.text:
                return response.text.strip()
            else:
                raise ValueError("Received empty or invalid response text from Gemini API.")

        except Exception as e:
            logger.error(f"Gemini API generation error: {e}")
            raise RuntimeError(f"Gemini API call failed: {e}") from e


class RAGPipeline:
    """
    End-to-end RAG Pipeline coordinating retrieval, context-building,
    Gemini generation, and programmatic citation construction.
    """

    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        generator: Optional[GeminiGenerator] = None,
        relevance_threshold: Optional[float] = None,
    ):
        self.retriever = retriever or DocumentRetriever()
        self.generator = generator or GeminiGenerator()
        self.relevance_threshold = (
            relevance_threshold
            if relevance_threshold is not None
            else float(os.getenv("RELEVANCE_THRESHOLD", DocumentRetriever.DEFAULT_SCORE_THRESHOLD))
        )

    def _build_context_prompt(
        self,
        question: str,
        retrieved_chunks: List[SearchResult],
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> str:
        """Constructs a grounded prompt containing retrieved runbook sections."""
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"--- SECTION {i} (Document: {chunk.source} | Original PDF Page: {chunk.page_number}) ---\n"
                f"{chunk.content}\n"
            )

        context_str = "\n".join(context_parts)

        # Include prior conversation turns if provided
        history_parts = []
        if conversation_history:
            history_parts.append("\nConversation History:")
            for msg in conversation_history[-4:]:
                history_parts.append(f"{msg.role.capitalize()}: {msg.content}")
            history_parts.append("\n")

        history_str = "\n".join(history_parts) if history_parts else ""

        prompt = (
            f"You are provided with the following authoritative excerpts from the TitanCloud SRE Incident Runbook:\n\n"
            f"{context_str}\n"
            f"{history_str}"
            f"Current Question: {question}\n\n"
            f"Instruction: Based strictly on the excerpts above, provide a comprehensive and accurate answer. "
            f"Explicitly cite the original PDF page number (e.g., [Page X]) for each fact or step mentioned. "
            f"If the excerpts do not contain the answer, reply exactly with: '{INSUFFICIENT_CONTEXT_MESSAGE}'."
        )
        return prompt

    def answer(
        self,
        question: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        top_k: int = 4,
        score_threshold: Optional[float] = None,
    ) -> RAGResponse:
        """
        Executes the full RAG pipeline for a user question.

        Args:
            question: User inquiry.
            conversation_history: Optional previous conversation messages.
            top_k: Number of relevant chunks to retrieve.
            score_threshold: Optional relevance threshold override.

        Returns:
            RAGResponse containing the answer, programmatic citations, confidence, and metadata.
        """
        if not question or not question.strip():
            return RAGResponse(
                answer="Please enter a valid question.",
                sources=[],
                is_sufficient=False,
            )

        # 1. Conversational Query Resolution for follow-ups
        resolved_query = self.retriever.resolve_conversational_query(
            question,
            conversation_history,
        )

        effective_threshold = (
            score_threshold
            if score_threshold is not None
            else self.relevance_threshold
        )

        # 2. Semantic & Hybrid Retrieval
        retrieved_chunks = self.retriever.retrieve(
            query=resolved_query,
            top_k=top_k,
            score_threshold=effective_threshold,
        )

        # 3. Handle Empty or Insufficient Context
        if not retrieved_chunks:
            logger.info(f"No chunks met relevance threshold ({effective_threshold}) for query '{resolved_query}'.")
            return RAGResponse(
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
                confidence=0.0,
                is_sufficient=False,
                query_used=resolved_query,
            )

        top_score = max(c.score for c in retrieved_chunks)

        # 4. Programmatic Citations (guarantees citations originate ONLY from actually retrieved chunks)
        sources: List[SourceCitation] = [
            SourceCitation(
                filename=chunk.source,
                page_number=chunk.page_number,
                text=chunk.content,
            )
            for chunk in retrieved_chunks
        ]

        # 5. Gemini Generation
        if not self.generator.is_available:
            # High-fidelity extractive fallback when GEMINI_API_KEY is not configured
            top_chunk = retrieved_chunks[0]
            top_text = top_chunk.content

            # Standard English stop words to isolate informative content keywords
            stop_words = {
                "what", "when", "where", "which", "who", "whom", "this", "that",
                "does", "have", "with", "from", "titancloud", "the", "and", "for",
                "are", "their", "our", "your", "its", "into", "onto", "about",
                "can", "could", "should", "would", "how", "why", "tell", "explain",
                "describe", "give", "please", "any", "all", "some", "such", "than",
                "then", "there", "here", "also", "been", "being", "were", "was",
                "out", "per", "one", "two", "use", "has", "not", "nor", "but"
            }

            query_words = [
                w.lower()
                for w in re.findall(r"\b[A-Za-z0-9_]{3,}\b", resolved_query)
                if w.lower() not in stop_words
            ]

            match_count = sum(1 for w in query_words if w in top_text.lower())
            if query_words and match_count == 0:
                logger.info(f"Top chunk has no keyword overlap with query keywords: {query_words}")
                return RAGResponse(
                    answer=INSUFFICIENT_CONTEXT_MESSAGE,
                    sources=[],
                    confidence=0.0,
                    is_sufficient=False,
                    query_used=resolved_query,
                )

            # Grounded extractive summary from retrieved text
            answer_text = (
                f"[Notice: GEMINI_API_KEY is not configured in environment]\n"
                f"According to the TitanCloud SRE Incident Runbook [Page {top_chunk.page_number}]:\n"
                f"{top_text.strip()}"
            )
            return RAGResponse(
                answer=answer_text,
                sources=sources,
                confidence=top_score,
                is_sufficient=True,
                query_used=resolved_query,
            )

        prompt = self._build_context_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
        )

        try:
            llm_answer = self.generator.generate(prompt=prompt)

            # Check if LLM determined the context is insufficient
            is_insufficient = (
                INSUFFICIENT_CONTEXT_MESSAGE.lower() in llm_answer.lower()
                or "does not contain sufficient information" in llm_answer.lower()
            )

            if is_insufficient:
                return RAGResponse(
                    answer=INSUFFICIENT_CONTEXT_MESSAGE,
                    sources=[],
                    confidence=top_score,
                    is_sufficient=False,
                    query_used=resolved_query,
                )

            return RAGResponse(
                answer=llm_answer,
                sources=sources,
                confidence=top_score,
                is_sufficient=True,
                query_used=resolved_query,
            )

        except Exception as e:
            logger.warning(f"Gemini API call encountered an issue ({e}). Providing direct verified runbook excerpt.")
            top_chunk = retrieved_chunks[0]
            answer_text = (
                f"[Notice: Gemini API connectivity issue ({type(e).__name__}). Showing authoritative runbook excerpt]\n"
                f"According to the TitanCloud SRE Incident Runbook [Page {top_chunk.page_number}]:\n"
                f"{top_chunk.content.strip()}"
            )
            return RAGResponse(
                answer=answer_text,
                sources=sources,
                confidence=top_score,
                is_sufficient=True,
                query_used=resolved_query,
            )
