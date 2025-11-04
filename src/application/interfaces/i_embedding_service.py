from abc import ABC, abstractmethod
from typing import List

class IEmbeddingService(ABC):
    """Interface for embedding service to convert text to vectors"""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Convert text to embedding vector.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts to embedding vectors.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        pass

