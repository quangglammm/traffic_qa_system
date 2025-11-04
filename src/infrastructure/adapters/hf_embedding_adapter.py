from typing import List
import logging
from src.application.interfaces.i_embedding_service import IEmbeddingService

logger = logging.getLogger(__name__)

class HFEmbeddingAdapter(IEmbeddingService):
    """Embedding service adapter using HuggingFace models"""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize HuggingFace Embedding Adapter.

        Args:
            model_name: Name of the HuggingFace embedding model to use
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"Initialized HF Embedding adapter with model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, using fallback")
            self.model = None
            self.model_name = model_name

    def embed_text(self, text: str) -> List[float]:
        """
        Convert text to embedding vector.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if self.model is None:
            # Fallback: return a dummy embedding
            logger.warning("Model not available, returning dummy embedding")
            return [0.0] * 384  # Default dimension for multilingual models

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return [0.0] * 384

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts to embedding vectors.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if self.model is None:
            # Fallback: return dummy embeddings
            logger.warning("Model not available, returning dummy embeddings")
            return [[0.0] * 384] * len(texts)

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            return [[0.0] * 384] * len(texts)

