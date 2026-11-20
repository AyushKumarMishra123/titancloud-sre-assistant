"""
TitanCloud SRE Incident Assistant - Embedding Generation Module
Uses Sentence Transformers (all-MiniLM-L6-v2) to generate normalized embeddings.
"""

import logging
import os
from typing import List, Optional, Union

# Set environment variables for macOS OpenMP and offline stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
# Set single thread to prevent OpenMP runtime conflicts on macOS arm64
torch.set_num_threads(1)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles embedding generation for documents and search queries using SentenceTransformers."""

    DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"
    DEFAULT_BATCH_SIZE: int = 32

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformers model (default: all-MiniLM-L6-v2).
            batch_size: Batch size for encoding (default: 32).
            device: Computing device ('cpu', 'cuda', 'mps', or None for auto).
        """
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL_NAME")
            or self.DEFAULT_MODEL_NAME
        )
        self.batch_size = (
            batch_size
            or int(os.getenv("EMBEDDING_BATCH_SIZE", self.DEFAULT_BATCH_SIZE))
        )
        self.device = device or "cpu"

        logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Safe attribute access for embedding dimension across sentence-transformers versions
            if hasattr(self.model, "get_embedding_dimension"):
                self.dimension = self.model.get_embedding_dimension()
            elif hasattr(self.model, "get_sentence_embedding_dimension"):
                self.dimension = self.model.get_sentence_embedding_dimension()
            else:
                self.dimension = 384
            logger.info(f"Loaded embedding model '{self.model_name}' with dimension {self.dimension}.")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
            raise RuntimeError(f"Could not initialize embedding model: {e}") from e

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Generate dense vector embeddings for a list of texts in batches.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding (overrides instance default if provided).
            normalize: Whether to normalize embeddings to unit L2 norm (enables cosine similarity via inner product).
            show_progress_bar: Whether to display a progress bar during generation.

        Returns:
            np.ndarray of shape (len(texts), dimension) and dtype float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        actual_batch_size = batch_size or self.batch_size

        try:
            # Use positional texts to avoid version discrepancies (sentences vs inputs)
            embeddings = self.model.encode(
                texts,
                batch_size=actual_batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            # Ensure float32 for FAISS compatibility
            embeddings = np.asarray(embeddings, dtype=np.float32)

            if normalize:
                # Extra verification of unit norm
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms

            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings for {len(texts)} texts: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def generate_query_embedding(
        self,
        query: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embedding for a single query text.

        Args:
            query: Query text string.
            normalize: Whether to normalize embedding to unit L2 norm.

        Returns:
            np.ndarray of shape (1, dimension) and dtype float32.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        query_clean = query.strip()
        embeddings = self.generate_embeddings(
            texts=[query_clean],
            batch_size=1,
            normalize=normalize,
            show_progress_bar=False,
        )
        return embeddings
