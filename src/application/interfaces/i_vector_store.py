from abc import ABC, abstractmethod
from typing import List, Tuple

class IVectorStore(ABC):
    """Interface for vector store to perform semantic search"""

    @abstractmethod
    def add_violations(self, violation_ids: List[str], embeddings: List[List[float]], descriptions: List[str]):
        """
        Add violation embeddings to the vector store.

        Args:
            violation_ids: List of violation IDs
            embeddings: List of embedding vectors corresponding to violation_ids
            descriptions: List of violation descriptions for indexing
        """
        pass

    @abstractmethod
    def search_similar(self, query_embedding: List[float], k: int = 5, min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """
        Search for similar violations using vector similarity.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of tuples (violation_id, similarity_score) sorted by similarity (descending)
        """
        pass

