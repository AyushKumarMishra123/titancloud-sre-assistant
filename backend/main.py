"""
TitanCloud SRE Incident Assistant - FastAPI REST Application
Provides production endpoints for health inspection, multi-turn chat,
document metadata, and vector index statistics.
"""

from contextlib import asynccontextmanager
import logging
import os
import uuid
from typing import Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.ingestion import compute_file_sha256
from backend.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ClearSessionRequest,
    ClearSessionResponse,
    DocumentMetadata,
    HealthResponse,
    RAGResponse,
    SearchResult,
    StatsResponse,
)
from backend.rag_pipeline import INSUFFICIENT_CONTEXT_MESSAGE, RAGPipeline
from backend.retrieval import DocumentRetriever

load_dotenv()
logger = logging.getLogger("titancloud.api")

# Singletons shared across request contexts to prevent redundant loading
retriever = DocumentRetriever()
rag_pipeline = RAGPipeline(retriever=retriever)

# In-memory session store for multi-turn conversational follow-ups
session_conversations: Dict[str, List[ChatMessage]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize models and vector store once."""
    logger.info("Initializing TitanCloud SRE Incident Assistant API...")
    try:
        loaded = retriever.load_index()
        if loaded:
            logger.info(f"Vector store successfully loaded ({len(retriever.vector_store.chunks)} chunks).")
        else:
            logger.warning("Vector store index was not found on startup. Execute 'python -m backend.ingestion' first.")
    except Exception as e:
        logger.error(f"Error initializing vector store on startup: {e}")
    yield
    logger.info("Shutting down TitanCloud SRE Incident Assistant API...")


app = FastAPI(
    title="TitanCloud SRE Incident Assistant API",
    description=(
        "Enterprise Site Reliability Engineering knowledge assistant providing grounded answers "
        "and verified page citations from the official TitanCloud SRE Incident Runbook."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware for local Streamlit frontend and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """
    Returns operational health status across all backend subsystems:
    - Backend application status
    - FAISS vector store status & chunk count
    - Dense embedding model configuration
    - Google Gemini LLM configuration status
    """
    if not retriever.is_loaded:
        retriever.load_index()

    is_loaded = retriever.is_loaded
    total_chunks = len(retriever.vector_store.chunks) if is_loaded else 0
    faiss_status = "active_and_loaded" if is_loaded else "not_loaded"

    gemini_configured = rag_pipeline.generator.is_available
    gemini_model = rag_pipeline.generator.model_name

    return HealthResponse(
        status="healthy",
        service="titancloud-sre-incident-assistant",
        version="1.0.0",
        faiss_status=faiss_status,
        vector_store_loaded=is_loaded,
        total_chunks=total_chunks,
        embedding_model=retriever.embedding_model_name,
        embedding_dimension=retriever.embedder.dimension,
        gemini_configured=gemini_configured,
        gemini_model=gemini_model,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat & RAG"])
def chat(request: ChatRequest):
    """
    Main conversational RAG endpoint for incident queries.
    - Resolves conversational follow-ups using session history.
    - Retrieves top-K relevant runbook chunks with hybrid exact matching.
    - Calls Gemini with authoritative system rules.
    - Returns grounded answer with programmatic PDF page citations.
    """
    if not retriever.is_loaded:
        if not retriever.load_index():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="FAISS vector store is not loaded. Please run the ingestion pipeline first.",
            )

    session_id = request.session_id or str(uuid.uuid4())

    # Retrieve prior conversation history for this session
    conversation_history = session_conversations.get(session_id, [])

    # If the client sent explicit history, use it
    if request.chat_history:
        conversation_history = request.chat_history

    try:
        rag_result: RAGResponse = rag_pipeline.answer(
            question=request.question,
            conversation_history=conversation_history,
            top_k=request.top_k,
        )

        retrieval_status = "success" if rag_result.is_sufficient else "no_context"

        # Update in-memory session history (retaining last 10 turns)
        updated_history = list(conversation_history)
        updated_history.append(ChatMessage(role="user", content=request.question))
        updated_history.append(ChatMessage(role="assistant", content=rag_result.answer))
        session_conversations[session_id] = updated_history[-10:]

        return ChatResponse(
            answer=rag_result.answer,
            sources=rag_result.sources,
            retrieval_status=retrieval_status,
            session_id=session_id,
            confidence=rag_result.confidence,
            is_sufficient=rag_result.is_sufficient,
            query_used=rag_result.query_used,
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while answering the query: {str(e)}",
        )


@app.post("/chat/clear", response_model=ClearSessionResponse, tags=["Chat & RAG"])
def clear_session(request: ClearSessionRequest):
    """Clears the conversational context for a given session ID."""
    session_id = request.session_id
    if session_id in session_conversations:
        del session_conversations[session_id]
        logger.info(f"Cleared session history for '{session_id}'.")
        return ClearSessionResponse(
            session_id=session_id,
            cleared=True,
            message="Conversation history cleared successfully.",
        )
    return ClearSessionResponse(
        session_id=session_id,
        cleared=False,
        message="Session ID not found or already empty.",
    )


@app.get("/stats", response_model=StatsResponse, tags=["Document & Index"])
def get_stats():
    """
    Returns actual vector index and corpus statistics:
    - Source document filename and SHA-256 hash
    - Total pages and total chunks indexed
    - Average, min, and max chunk character lengths
    - Embedding model and vector dimensionality
    - Persistent index directory and creation timestamp
    """
    if not retriever.is_loaded:
        retriever.load_index()

    manifest = retriever.vector_store.manifest or {}
    chunks = retriever.vector_store.chunks

    chunk_lengths = [len(c.content) for c in chunks] if chunks else [0]
    total_pages = manifest.get("total_pages", 50)
    pdf_path = manifest.get("pdf_path", "data/titancloud_sre_incident_runbook_50p.pdf")
    pdf_sha256 = manifest.get("pdf_sha256", "")

    if not pdf_sha256 and os.path.exists(pdf_path):
        pdf_sha256 = compute_file_sha256(pdf_path)

    return StatsResponse(
        indexed_document=os.path.basename(pdf_path),
        total_pages=total_pages,
        total_chunks=len(chunks),
        avg_chunk_length=float(np.mean(chunk_lengths)) if chunks else 0.0,
        min_chunk_length=int(min(chunk_lengths)) if chunks else 0,
        max_chunk_length=int(max(chunk_lengths)) if chunks else 0,
        chunk_size=manifest.get("chunk_size", 900),
        chunk_overlap=manifest.get("chunk_overlap", 120),
        embedding_model=manifest.get("embedding_model", retriever.embedding_model_name),
        embedding_dimension=manifest.get("embedding_dimension", 384),
        vector_database_status="active" if retriever.is_loaded else "offline",
        index_directory=retriever.index_dir,
        pdf_sha256=pdf_sha256,
        created_at=manifest.get("created_at"),
    )


@app.get("/documents", response_model=DocumentMetadata, tags=["Document & Index"])
def get_documents_metadata():
    """
    Returns metadata for the single fixed runbook document.
    Note: Document uploads and document replacements are disabled by design.
    """
    pdf_path = "data/titancloud_sre_incident_runbook_50p.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Authoritative runbook document not found at '{pdf_path}'.",
        )

    file_size = os.path.getsize(pdf_path)
    file_sha256 = compute_file_sha256(pdf_path)

    if not retriever.is_loaded:
        retriever.load_index()

    manifest = retriever.vector_store.manifest or {}
    total_chunks = len(retriever.vector_store.chunks) if retriever.is_loaded else 100

    return DocumentMetadata(
        filename=os.path.basename(pdf_path),
        file_path=pdf_path,
        file_size_bytes=file_size,
        pdf_sha256=file_sha256,
        total_pages=manifest.get("total_pages", 50),
        total_chunks=total_chunks,
        chunk_size=manifest.get("chunk_size", 900),
        chunk_overlap=manifest.get("chunk_overlap", 120),
        description="Official TitanCloud SRE Incident Response Runbook (50 Pages Reference Corpus)",
        is_read_only=True,
    )


@app.get("/search", response_model=List[SearchResult], tags=["Chat & RAG"])
def search_runbook(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=4, ge=1, le=20, description="Number of results to retrieve"),
):
    """Developer endpoint: performs direct semantic search against the FAISS index."""
    if not retriever.is_loaded:
        if not retriever.load_index():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store index is not loaded. Please execute ingestion pipeline first.",
            )
    try:
        return retriever.retrieve(query=q, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
