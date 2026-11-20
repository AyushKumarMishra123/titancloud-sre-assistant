"""
TitanCloud SRE Incident Assistant - Pydantic Data Models
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """Represents a single extracted page from a document."""
    page_number: int = Field(..., description="1-indexed page number from the original document")
    text: str = Field(..., description="Cleaned text content of the page")
    source: str = Field(..., description="Source filename or file path")
    char_count: int = Field(..., description="Character count of cleaned text")


class DocumentChunk(BaseModel):
    """Represents a chunk of text derived from a document page."""
    chunk_id: str = Field(..., description="Unique chunk identifier, e.g., chunk_p1_0")
    source: str = Field(..., description="Source document filename")
    page_number: int = Field(..., description="1-indexed page number where this chunk originates")
    content: str = Field(..., description="Text content of the chunk")
    char_count: int = Field(..., description="Character count of the chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional chunk metadata")


class IngestionStats(BaseModel):
    """Detailed statistics for the ingestion process."""
    total_pages_in_pdf: int
    extracted_pages_count: int
    total_chunks: int
    embedding_model: str
    embedding_dimension: int
    pdf_path: str
    pdf_sha256: str
    index_directory: str
    chunk_size: int
    chunk_overlap: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    status: str = "completed"


class SearchResult(BaseModel):
    """Single retrieval result from the vector store."""
    chunk_id: str
    source: str
    page_number: int
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceCitation(BaseModel):
    """Authoritative source citation for an answer derived directly from chunk metadata."""
    filename: str = Field(..., description="Original filename of the source document")
    page_number: int = Field(..., description="Original 1-indexed PDF page number")
    text: str = Field(..., description="Retrieved source content from the page")


class ChatMessage(BaseModel):
    """Single turn in a conversation."""
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    """Request payload for the POST /chat endpoint."""
    question: str = Field(..., min_length=1, max_length=1500, description="User question or incident query")
    session_id: Optional[str] = Field(default=None, description="Optional unique session identifier")
    chat_history: Optional[List[ChatMessage]] = Field(
        default_factory=list, description="Prior conversation turns for follow-up resolution"
    )
    top_k: int = Field(default=4, ge=1, le=10, description="Number of context chunks to retrieve")


class ChatResponse(BaseModel):
    """Response payload for the POST /chat endpoint."""
    answer: str = Field(..., description="Grounded answer synthesized from the runbook")
    sources: List[SourceCitation] = Field(
        default_factory=list, description="Verified source citations with original PDF page numbers"
    )
    retrieval_status: str = Field(default="success", description="Status of retrieval ('success', 'no_context')")
    session_id: Optional[str] = Field(default=None, description="Session ID associated with the interaction")
    confidence: Optional[float] = Field(default=None, description="Top retrieval similarity score")
    is_sufficient: bool = Field(default=True, description="Whether the runbook context was sufficient")
    query_used: Optional[str] = Field(default=None, description="Resolved query string used for retrieval")


class RAGRequest(BaseModel):
    """Internal request payload for RAG question answering."""
    question: str = Field(..., min_length=1, description="User question")
    conversation_history: List[ChatMessage] = Field(
        default_factory=list, description="Prior conversation turns for follow-up resolution"
    )
    top_k: int = Field(default=4, ge=1, le=10, description="Number of context chunks to retrieve")
    score_threshold: Optional[float] = Field(
        default=None, description="Optional minimum similarity score threshold"
    )


class RAGResponse(BaseModel):
    """Structured response from the RAG pipeline."""
    answer: str = Field(..., description="Grounded answer synthesized from the runbook")
    sources: List[SourceCitation] = Field(
        default_factory=list, description="Verified source citations with original PDF page numbers"
    )
    confidence: Optional[float] = Field(
        default=None, description="Top retrieval similarity score"
    )
    is_sufficient: bool = Field(
        default=True, description="Whether the runbook contained sufficient information"
    )
    query_used: Optional[str] = Field(
        default=None, description="Resolved query string used for retrieval"
    )


class HealthResponse(BaseModel):
    """System health response model for GET /health."""
    status: str = "healthy"
    service: str = "titancloud-sre-incident-assistant"
    version: str = "1.0.0"
    faiss_status: str
    vector_store_loaded: bool
    total_chunks: int
    embedding_model: str
    embedding_dimension: int
    gemini_configured: bool
    gemini_model: str


class StatsResponse(BaseModel):
    """System statistics response model for GET /stats."""
    indexed_document: str
    total_pages: int
    total_chunks: int
    avg_chunk_length: float
    min_chunk_length: int
    max_chunk_length: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    embedding_dimension: int
    vector_database_status: str
    index_directory: str
    pdf_sha256: str
    created_at: Optional[str] = None


class DocumentMetadata(BaseModel):
    """Metadata response for GET /documents."""
    filename: str
    file_path: str
    file_size_bytes: int
    pdf_sha256: str
    total_pages: int
    total_chunks: int
    chunk_size: int
    chunk_overlap: int
    description: str
    is_read_only: bool = True


class ClearSessionRequest(BaseModel):
    """Request payload to clear conversation history."""
    session_id: str = Field(..., min_length=1, description="Unique session ID to clear")


class ClearSessionResponse(BaseModel):
    """Response payload after clearing conversation history."""
    session_id: str
    cleared: bool
    message: str
