from typing import List, Tuple
import logging
import uuid
from src.application.interfaces.i_vector_store import IVectorStore

logger = logging.getLogger(__name__)

class ChromaVSAdapter(IVectorStore):
    """Vector store adapter using ChromaDB"""

    def __init__(self, collection_name: str = "traffic_violations", persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB Vector Store Adapter.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the ChromaDB database
        """
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )

            logger.info(f"Initialized ChromaDB adapter with collection: {collection_name}")
        except ImportError:
            logger.warning("chromadb not installed, using in-memory fallback")
            self.client = None
            self.collection = None
            self._in_memory_store = {}  # Fallback: simple in-memory store

    def add_violations(self, violation_ids: List[str], embeddings: List[List[float]], descriptions: List[str]):
        """
        Add violation embeddings to the vector store.

        Args:
            violation_ids: List of violation IDs
            embeddings: List of embedding vectors corresponding to violation_ids
            descriptions: List of violation descriptions for indexing
        """
        if not violation_ids or not embeddings or not descriptions:
            logger.warning("Empty input data for add_violations")
            return

        if len(violation_ids) != len(embeddings) or len(violation_ids) != len(descriptions):
            raise ValueError("violation_ids, embeddings, and descriptions must have the same length")

        try:
            if self.collection is not None:
                # Use ChromaDB
                # Generate unique IDs for each document
                ids = [f"violation_{vid}" for vid in violation_ids]

                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=descriptions,
                    metadatas=[{"violation_id": vid} for vid in violation_ids]
                )

                logger.info(f"Added {len(violation_ids)} violations to ChromaDB")
            else:
                # Fallback: store in memory
                for vid, emb, desc in zip(violation_ids, embeddings, descriptions):
                    self._in_memory_store[vid] = {
                        "embedding": emb,
                        "description": desc
                    }
                logger.info(f"Added {len(violation_ids)} violations to in-memory store (fallback)")
        except Exception as e:
            logger.error(f"Error adding violations to vector store: {e}")
            raise

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
        if not query_embedding:
            return []

        try:
            if self.collection is not None:
                # Use ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k
                )

                # Extract violation IDs and similarity scores
                violation_results = []
                if results["ids"] and len(results["ids"]) > 0:
                    ids = results["ids"][0]
                    metadatas = results["metadatas"][0]
                    distances = results["distances"][0] if "distances" in results else None

                    for idx, doc_id in enumerate(ids):
                        if metadatas and idx < len(metadatas):
                            violation_id = metadatas[idx].get("violation_id", doc_id.replace("violation_", ""))

                            # Convert distance to similarity (ChromaDB uses cosine distance)
                            # Cosine distance = 1 - cosine similarity
                            # So similarity = 1 - distance
                            if distances is not None and idx < len(distances):
                                similarity = 1.0 - distances[idx]
                            else:
                                similarity = 1.0  # Default if no distance available

                            if similarity >= min_similarity:
                                violation_results.append((violation_id, similarity))

                # Sort by similarity (descending)
                violation_results.sort(key=lambda x: x[1], reverse=True)
                return violation_results
            else:
                # Fallback: simple cosine similarity in memory
                import numpy as np

                results = []
                for vid, data in self._in_memory_store.items():
                    emb = data["embedding"]
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, emb)
                    if similarity >= min_similarity:
                        results.append((vid, similarity))

                # Sort by similarity and return top k
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:k]
        except Exception as e:
            logger.error(f"Error searching similar violations: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0

