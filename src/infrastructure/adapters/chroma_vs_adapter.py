from typing import List, Tuple, Optional, Dict, Any
import logging
from src.application.interfaces.i_vector_store import IVectorStore

logger = logging.getLogger(__name__)


class ChromaVSAdapter(IVectorStore):
    """Optimized ChromaDB adapter for traffic violation semantic search"""

    def __init__(
        self,
        collection_name: str = "traffic_violations",
        persist_directory: str = "./chroma_db",
    ):
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},  # Best for semantic search
            )
            self._use_chroma = True
            logger.info(
                f"ChromaDB initialized: {collection_name} at {persist_directory}"
            )

        except ImportError as e:
            logger.warning("chromadb not installed. Falling back to in-memory store.")
            self._use_chroma = False
            self._in_memory: Dict[str, Dict[str, Any]] = {}
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Add violations with rich metadata (vehicle_type, version, etc.)
        """
        if not ids or not embeddings or not documents:
            logger.warning("Empty data passed to add_documents")
            return

        if len(ids) != len(embeddings) or len(ids) != len(documents):
            raise ValueError("ids, embeddings, and documents must have the same length")

        if metadatas is None:
            metadatas = [{} for _ in ids]

        if len(metadatas) != len(ids):
            raise ValueError("metadatas must match the number of documents")

        try:
            if self._use_chroma:
                chroma_ids = [
                    f"doc_{i}_{vid}" for i, vid in enumerate(ids)
                ]  # Unique + readable

                self.collection.add(
                    ids=chroma_ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=[
                        {"violation_id": vid, **meta}
                        for vid, meta in zip(ids, metadatas)
                    ],
                )
                logger.info(f"Added {len(ids)} documents to ChromaDB collection")
            else:
                for vid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
                    self._in_memory[vid] = {
                        "embedding": emb,
                        "document": doc,
                        "metadata": meta,
                    }
                logger.info(f"Added {len(ids)} documents to in-memory store (fallback)")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}", exc_info=True)
            raise

    def search_similar(
        self,
        query_embedding: List[float],
        k: int = 10,
        min_similarity: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search similar violations with metadata filtering (e.g., vehicle_type)

        Returns:
            List of (violation_id, similarity_score, metadata)
        """
        if not query_embedding:
            return []

        try:
            if self._use_chroma:
                where_clause = None
                if filter_metadata:
                    # Support exact match on metadata fields
                    where_clause = {
                        k: v for k, v in filter_metadata.items() if v is not None
                    }

                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k * 2,  # Get more, then filter/rerank
                    where=where_clause,
                    include=["metadatas", "distances", "documents"],
                )

                hits = []
                if results["ids"] and results["ids"][0]:
                    for i, chroma_id in enumerate(results["ids"][0]):
                        distance = results["distances"][0][i]
                        similarity = max(
                            0.0, 1.0 - distance
                        )  # cosine distance → similarity

                        if similarity < min_similarity:
                            continue

                        metadata = results["metadatas"][0][i]
                        violation_id = metadata.get("violation_id", chroma_id)

                        hits.append((violation_id, round(similarity, 4), metadata))

                # Sort by similarity descending
                hits.sort(key=lambda x: x[1], reverse=True)
                return hits[:k]

            else:
                # In-memory fallback
                import numpy as np

                q = np.array(query_embedding)
                hits = []

                for vid, data in self._in_memory.items():
                    if filter_metadata:
                        if not all(
                            data["metadata"].get(k) == v
                            for k, v in filter_metadata.items()
                        ):
                            continue

                    emb = np.array(data["embedding"])
                    sim = float(
                        np.dot(q, emb)
                        / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-10)
                    )
                    if sim >= min_similarity:
                        hits.append((vid, round(sim, 4), data["metadata"]))

                hits.sort(key=lambda x: x[1], reverse=True)
                return hits[:k]

        except Exception as e:
            logger.error(f"Error during vector search: {e}", exc_info=True)
            return []

    def reset(self):
        """Clear all data (useful for testing)"""
        try:
            if self._use_chroma:
                self.client.reset()
                logger.info("ChromaDB reset successfully")
            else:
                self._in_memory.clear()
        except Exception as e:
            logger.error(f"Error resetting vector store: {e}")

    def count(self) -> int:
        """Return number of stored documents"""
        try:
            if self._use_chroma:
                return self.collection.count()
            else:
                return len(self._in_memory)
        except:
            return 0
