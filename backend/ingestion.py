"""
TitanCloud SRE Incident Assistant - Document Ingestion Pipeline
Handles PDF extraction (PyMuPDF), recursive character chunking,
embedding generation, persistent FAISS indexing, and PDF change detection via SHA-256.
"""

import os
# Prevent OpenMP runtime collision on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np
import pymupdf

from backend.embeddings import EmbeddingGenerator
from backend.models import DocumentChunk, DocumentPage, IngestionStats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: str) -> str:
    """Compute the SHA-256 checksum of a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found for hashing: {file_path}")

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class PDFExtractor:
    """Extracts text and page-level metadata from PDF documents using PyMuPDF."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF document does not exist at path: {pdf_path}")

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans extracted PDF text while strictly preserving technical terms,
        headings, bullet points, numbers, and paragraph structure.
        """
        if not text:
            return ""

        # Normalize carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace non-breaking spaces and exotic whitespace with standard space
        text = text.replace("\xa0", " ").replace("\u200b", "")

        # Collapse trailing horizontal whitespace on lines
        lines = [re.sub(r"[ \t]+$", "", line) for line in text.split("\n")]

        # Collapse sequences of 3 or more blank lines into a double newline (paragraph break)
        cleaned_text = "\n".join(lines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

        return cleaned_text.strip()

    def extract_pages(self) -> List[DocumentPage]:
        """
        Extracts all pages from the PDF document.
        Preserves 1-indexed page numbers and source provenance.

        Returns:
            List of DocumentPage objects for all valid pages.
        """
        doc = None
        extracted_pages: List[DocumentPage] = []
        source_name = os.path.basename(self.pdf_path)

        try:
            doc = pymupdf.open(self.pdf_path)
            total_pages = len(doc)
            logger.info(f"Opened PDF '{self.pdf_path}' ({total_pages} total pages).")

            for page_index in range(total_pages):
                page_number = page_index + 1  # 1-indexed
                try:
                    page = doc[page_index]
                    raw_text = page.get_text("text")
                    cleaned = self.clean_text(raw_text)

                    if not cleaned:
                        logger.warning(f"Page {page_number} in '{source_name}' contains no extractable text.")
                        # Still record the page structure to preserve boundaries and auditability
                        cleaned = ""

                    doc_page = DocumentPage(
                        page_number=page_number,
                        text=cleaned,
                        source=source_name,
                        char_count=len(cleaned),
                    )
                    extracted_pages.append(doc_page)

                except Exception as page_err:
                    logger.error(f"Error extracting text from page {page_number}: {page_err}")
                    # Include empty placeholder page to not lose page number sequence
                    extracted_pages.append(
                        DocumentPage(
                            page_number=page_number,
                            text="",
                            source=source_name,
                            char_count=0,
                        )
                    )

            logger.info(f"Successfully extracted {len(extracted_pages)} pages from '{self.pdf_path}'.")
            return extracted_pages

        except Exception as e:
            logger.error(f"Fatal error reading PDF '{self.pdf_path}': {e}")
            raise RuntimeError(f"Could not read PDF '{self.pdf_path}': {e}") from e
        finally:
            if doc is not None:
                doc.close()


class RecursiveCharacterTextSplitter:
    """
    Splits text recursively using character delimiters in order of precedence:
    Paragraphs (\\n\\n), Lines (\\n), Sentences (. ), Words (' '), and Characters ('').
    """

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        separators: Optional[List[str]] = None,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursive helper to break text into chunks under chunk_size."""
        final_chunks: List[str] = []

        # Find the first separator present in the text
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split text by chosen separator
        splits = text.split(separator) if separator != "" else list(text)

        good_splits: List[str] = []
        for s in splits:
            if s:
                good_splits.append(s)

        # Merge splits up to chunk_size with chunk_overlap
        merged_chunks = self._merge_splits(good_splits, separator, new_separators)
        final_chunks.extend(merged_chunks)
        return final_chunks

    def _merge_splits(
        self,
        splits: List[str],
        separator: str,
        new_separators: List[str],
    ) -> List[str]:
        """Merge smaller splits into chunks respecting chunk_size and chunk_overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for s in splits:
            # If an individual split is larger than chunk_size, recursively split it with finer separators
            if len(s) > self.chunk_size:
                if current_doc:
                    doc_str = separator.join(current_doc).strip()
                    if doc_str:
                        docs.append(doc_str)
                    current_doc = []
                    total = 0

                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    docs.extend(sub_splits)
                else:
                    # Character slice fallback
                    for start in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        sub_chunk = s[start : start + self.chunk_size].strip()
                        if sub_chunk:
                            docs.append(sub_chunk)
                continue

            # Length of adding this split
            add_len = len(s) + (len(separator) if current_doc else 0)
            if total + add_len > self.chunk_size:
                if current_doc:
                    doc_str = separator.join(current_doc).strip()
                    if doc_str:
                        docs.append(doc_str)

                    # Keep overlap for next chunk
                    overlap_doc: List[str] = []
                    overlap_len = 0
                    for item in reversed(current_doc):
                        item_len = len(item) + (len(separator) if overlap_doc else 0)
                        if overlap_len + item_len <= self.chunk_overlap:
                            overlap_doc.insert(0, item)
                            overlap_len += item_len
                        else:
                            break
                    current_doc = overlap_doc
                    total = overlap_len

            current_doc.append(s)
            total += len(s) + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc_str = separator.join(current_doc).strip()
            if doc_str:
                docs.append(doc_str)

        return docs

    def split_pages(self, pages: List[DocumentPage]) -> List[DocumentChunk]:
        """
        Split a list of DocumentPages into DocumentChunks.
        Preserves original page boundaries: text is split per-page so every chunk
        is strictly linked to its original 1-indexed page number.

        Args:
            pages: List of DocumentPage objects.

        Returns:
            List of DocumentChunk objects with unique chunk_ids and complete metadata.
        """
        all_chunks: List[DocumentChunk] = []

        for page in pages:
            if not page.text or not page.text.strip():
                continue

            raw_chunks = self._split_text(page.text, self.separators)

            for chunk_idx, content in enumerate(raw_chunks):
                clean_content = content.strip()
                if not clean_content:
                    continue

                chunk_id = f"chunk_p{page.page_number}_{chunk_idx + 1}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    source=page.source,
                    page_number=page.page_number,
                    content=clean_content,
                    char_count=len(clean_content),
                    metadata={
                        "source": page.source,
                        "page_number": page.page_number,
                        "chunk_index": chunk_idx + 1,
                        "total_page_chars": page.char_count,
                        "char_count": len(clean_content),
                    },
                )
                all_chunks.append(chunk)

        logger.info(f"Split {len(pages)} pages into {len(all_chunks)} chunks.")
        return all_chunks


class FAISSVectorStore:
    """
    Manages FAISS vector index, chunk metadata storage, persistence,
    and PDF change detection via SHA-256 file hashing.
    """

    INDEX_FILENAME = "index.faiss"
    CHUNKS_FILENAME = "chunks.json"
    MANIFEST_FILENAME = "index_manifest.json"

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []
        self.manifest: Dict[str, Any] = {}

    def is_index_valid(
        self,
        pdf_path: str,
        expected_model: str,
        expected_chunk_size: int,
        expected_chunk_overlap: int,
    ) -> Tuple[bool, str]:
        """
        Checks whether the existing local index is valid and up-to-date.

        Returns:
            (is_valid: bool, reason: str)
        """
        index_file = os.path.join(self.index_dir, self.INDEX_FILENAME)
        chunks_file = os.path.join(self.index_dir, self.CHUNKS_FILENAME)
        manifest_file = os.path.join(self.index_dir, self.MANIFEST_FILENAME)

        if not (os.path.exists(index_file) and os.path.exists(chunks_file) and os.path.exists(manifest_file)):
            return False, "Index files do not exist."

        # Check PDF existence and hash
        if not os.path.exists(pdf_path):
            return False, f"Source PDF not found at {pdf_path}."

        try:
            current_pdf_hash = compute_file_sha256(pdf_path)
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            if manifest.get("pdf_sha256") != current_pdf_hash:
                return False, f"PDF SHA-256 hash changed (cached: {manifest.get('pdf_sha256')[:8]}..., current: {current_pdf_hash[:8]}...)."

            if manifest.get("embedding_model") != expected_model:
                return False, f"Embedding model changed (cached: {manifest.get('embedding_model')}, current: {expected_model})."

            if (
                manifest.get("chunk_size") != expected_chunk_size
                or manifest.get("chunk_overlap") != expected_chunk_overlap
            ):
                return False, "Chunking configuration changed."

            # Verify index and chunk count match
            test_index = faiss.read_index(index_file)
            with open(chunks_file, "r", encoding="utf-8") as f:
                loaded_chunks = json.load(f)

            if test_index.ntotal != len(loaded_chunks):
                return False, f"Index vector count ({test_index.ntotal}) does not match chunk count ({len(loaded_chunks)})."

            return True, "Index is valid and up-to-date."

        except Exception as e:
            return False, f"Error validating existing index: {e}"

    def build_index(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
        pdf_path: str,
        pdf_sha256: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        total_pages: int,
    ) -> None:
        """
        Constructs a FAISS IndexFlatIP (inner product on normalized vectors = cosine similarity)
        and populates metadata.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) != number of embeddings ({len(embeddings)}).")

        dimension = embeddings.shape[1]
        logger.info(f"Building FAISS IndexFlatIP with {len(chunks)} vectors (dimension {dimension})...")

        # Create IndexFlatIP (inner product = cosine similarity for unit vectors)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        self.index = index
        self.chunks = chunks
        self.manifest = {
            "pdf_path": pdf_path,
            "pdf_sha256": pdf_sha256,
            "embedding_model": embedding_model,
            "embedding_dimension": dimension,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def save(self) -> None:
        """Persists the FAISS index, chunk metadata, and manifest to disk."""
        if self.index is None or not self.chunks:
            raise ValueError("Cannot save empty vector store.")

        os.makedirs(self.index_dir, exist_ok=True)
        index_file = os.path.join(self.index_dir, self.INDEX_FILENAME)
        chunks_file = os.path.join(self.index_dir, self.CHUNKS_FILENAME)
        manifest_file = os.path.join(self.index_dir, self.MANIFEST_FILENAME)

        logger.info(f"Writing FAISS index to '{index_file}'...")
        faiss.write_index(self.index, index_file)

        logger.info(f"Writing {len(self.chunks)} chunk metadata records to '{chunks_file}'...")
        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Writing manifest to '{manifest_file}'...")
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

        logger.info("Vector store persisted successfully.")

    def load(self) -> None:
        """Loads existing index, chunk metadata, and manifest from disk."""
        index_file = os.path.join(self.index_dir, self.INDEX_FILENAME)
        chunks_file = os.path.join(self.index_dir, self.CHUNKS_FILENAME)
        manifest_file = os.path.join(self.index_dir, self.MANIFEST_FILENAME)

        if not (os.path.exists(index_file) and os.path.exists(chunks_file) and os.path.exists(manifest_file)):
            raise FileNotFoundError(f"Missing index files in '{self.index_dir}'.")

        logger.info(f"Loading FAISS index from '{index_file}'...")
        self.index = faiss.read_index(index_file)

        logger.info(f"Loading chunks from '{chunks_file}'...")
        with open(chunks_file, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        self.chunks = [DocumentChunk(**item) for item in raw_chunks]

        with open(manifest_file, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors and {len(self.chunks)} chunks.")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Searches the FAISS index using a normalized query vector.

        Returns:
            List of (DocumentChunk, similarity_score) tuples sorted in descending order of score.
        """
        if self.index is None:
            raise RuntimeError("FAISS index has not been initialized or loaded.")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores, indices = self.index.search(query_vector.astype(np.float32), top_k)

        results: List[Tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results


class IngestionPipeline:
    """Orchestrates PDF reading, extraction, chunking, embedding, indexing, and persistence."""

    def __init__(
        self,
        pdf_path: Optional[str] = None,
        index_dir: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model_name: Optional[str] = None,
    ):
        self.pdf_path = (
            pdf_path
            or os.getenv("PDF_PATH")
            or "data/titancloud_sre_incident_runbook_50p.pdf"
        )
        self.index_dir = (
            index_dir
            or os.getenv("VECTOR_STORE_DIR")
            or "vector_store/faiss_index"
        )
        self.chunk_size = (
            chunk_size
            or int(os.getenv("CHUNK_SIZE", 900))
        )
        self.chunk_overlap = (
            chunk_overlap
            or int(os.getenv("CHUNK_OVERLAP", 120))
        )
        self.embedding_model_name = (
            embedding_model_name
            or os.getenv("EMBEDDING_MODEL_NAME")
            or EmbeddingGenerator.DEFAULT_MODEL_NAME
        )

    def run(self, force_rebuild: bool = False) -> IngestionStats:
        """
        Executes the ingestion pipeline.

        Args:
            force_rebuild: If True, forces re-extraction, re-embedding, and index rebuilding.

        Returns:
            IngestionStats instance containing execution details.
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"Required PDF runbook was not found at '{self.pdf_path}'.")

        pdf_sha256 = compute_file_sha256(self.pdf_path)
        vector_store = FAISSVectorStore(self.index_dir)

        if not force_rebuild:
            is_valid, reason = vector_store.is_index_valid(
                pdf_path=self.pdf_path,
                expected_model=self.embedding_model_name,
                expected_chunk_size=self.chunk_size,
                expected_chunk_overlap=self.chunk_overlap,
            )
            if is_valid:
                logger.info(f"Existing index in '{self.index_dir}' is valid ({reason}). Loading...")
                vector_store.load()
                manifest = vector_store.manifest
                chunk_lengths = [len(c.content) for c in vector_store.chunks]
                return IngestionStats(
                    total_pages_in_pdf=manifest.get("total_pages", 0),
                    extracted_pages_count=manifest.get("total_pages", 0),
                    total_chunks=len(vector_store.chunks),
                    embedding_model=manifest.get("embedding_model", self.embedding_model_name),
                    embedding_dimension=manifest.get("embedding_dimension", 384),
                    pdf_path=self.pdf_path,
                    pdf_sha256=pdf_sha256,
                    index_directory=self.index_dir,
                    chunk_size=manifest.get("chunk_size", self.chunk_size),
                    chunk_overlap=manifest.get("chunk_overlap", self.chunk_overlap),
                    avg_chunk_size=float(np.mean(chunk_lengths)) if chunk_lengths else 0.0,
                    min_chunk_size=int(min(chunk_lengths)) if chunk_lengths else 0,
                    max_chunk_size=int(max(chunk_lengths)) if chunk_lengths else 0,
                    status="reused_existing_index",
                )
            else:
                logger.info(f"Rebuilding index because: {reason}")

        # 1. PDF Extraction
        logger.info(f"Extracting pages from '{self.pdf_path}'...")
        extractor = PDFExtractor(self.pdf_path)
        extracted_pages = extractor.extract_pages()
        total_pages = len(extracted_pages)
        non_empty_pages = sum(1 for p in extracted_pages if p.text.strip())

        # 2. Chunking
        logger.info(f"Chunking {total_pages} pages (size={self.chunk_size}, overlap={self.chunk_overlap})...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_pages(extracted_pages)

        if not chunks:
            raise RuntimeError("No chunks were generated from the extracted PDF pages.")

        chunk_texts = [c.content for c in chunks]
        chunk_lengths = [len(c) for c in chunk_texts]

        # 3. Embedding Generation
        logger.info(f"Generating embeddings for {len(chunks)} chunks with '{self.embedding_model_name}'...")
        embedder = EmbeddingGenerator(model_name=self.embedding_model_name)
        embeddings = embedder.generate_embeddings(
            texts=chunk_texts,
            normalize=True,
            show_progress_bar=False,
        )

        # 4. FAISS Index Building
        vector_store.build_index(
            chunks=chunks,
            embeddings=embeddings,
            pdf_path=self.pdf_path,
            pdf_sha256=pdf_sha256,
            embedding_model=self.embedding_model_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            total_pages=total_pages,
        )

        # 5. Persistence
        vector_store.save()

        stats = IngestionStats(
            total_pages_in_pdf=total_pages,
            extracted_pages_count=non_empty_pages,
            total_chunks=len(chunks),
            embedding_model=self.embedding_model_name,
            embedding_dimension=embeddings.shape[1],
            pdf_path=self.pdf_path,
            pdf_sha256=pdf_sha256,
            index_directory=self.index_dir,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            avg_chunk_size=float(np.mean(chunk_lengths)),
            min_chunk_size=int(min(chunk_lengths)),
            max_chunk_size=int(max(chunk_lengths)),
            status="newly_indexed",
        )
        return stats


def print_ingestion_summary(stats: IngestionStats) -> None:
    """Displays a clean developer-facing summary table of the ingestion results."""
    border = "=" * 70
    print("\n" + border)
    print("      TITANCLOUD SRE RUNBOOK — INGESTION PIPELINE SUMMARY")
    print(border)
    print(f" Source PDF:          {stats.pdf_path}")
    print(f" PDF SHA-256:         {stats.pdf_sha256}")
    print(f" Total PDF Pages:     {stats.total_pages_in_pdf}")
    print(f" Pages with Text:     {stats.extracted_pages_count}")
    print(f" Total Chunks:        {stats.total_chunks}")
    print(f" Chunk Configuration: size={stats.chunk_size}, overlap={stats.chunk_overlap}")
    print(f" Avg Chunk Length:    {stats.avg_chunk_size:.1f} characters")
    print(f" Min / Max Chunk:     {stats.min_chunk_size} / {stats.max_chunk_size} characters")
    print(f" Embedding Model:     {stats.embedding_model}")
    print(f" Embedding Dimension: {stats.embedding_dimension}")
    print(f" FAISS Index Dir:     {stats.index_directory}")
    print(f" Ingestion Status:    {stats.status.upper()}")
    print(border + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TitanCloud SRE Runbook Document Ingestion")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF runbook file")
    parser.add_argument("--index-dir", type=str, default=None, help="Directory to store FAISS index")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in characters")
    parser.add_argument("--force", action="store_true", help="Force rebuild the index from scratch")

    args = parser.parse_args()

    pipeline = IngestionPipeline(
        pdf_path=args.pdf,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    results = pipeline.run(force_rebuild=args.force)
    print_ingestion_summary(results)
