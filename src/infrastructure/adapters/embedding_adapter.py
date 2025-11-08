from typing import List, Optional
import logging
from src.application.interfaces.i_embedding_service import IEmbeddingService

logger = logging.getLogger(__name__)


class OpenAIEmbeddingBackend(IEmbeddingService):
    """Backend for OpenAI-compatible embedding APIs"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, model_name: str = "Vietnamese_Embedding"):
        """
        Initialize OpenAI-compatible embedding backend.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for API (None for official OpenAI, custom for others)
            model_name: Embedding model name to use
        """
        try:
            from openai import OpenAI
            
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model_name = model_name
            logger.info(f"Initialized OpenAI-compatible embedding backend with model: {model_name}")
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text using OpenAI-compatible API"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error embedding text with OpenAI-compatible API: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using OpenAI-compatible API"""
        if not texts:
            return []
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error embedding batch with OpenAI-compatible API: {e}")
            raise


class HuggingFaceEmbeddingBackend(IEmbeddingService):
    """Backend for local HuggingFace embedding models"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize HuggingFace embedding backend.
        
        Args:
            model_name: Name of the HuggingFace embedding model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"Initialized HuggingFace embedding backend with model: {model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text using HuggingFace model"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding text with HuggingFace: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using HuggingFace model"""
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding batch with HuggingFace: {e}")
            raise


class EmbeddingAdapter(IEmbeddingService):
    """Universal Embedding Adapter supporting multiple backends"""

    def __init__(
        self,
        backend_type: str = "huggingface",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize Universal Embedding Adapter.

        Args:
            backend_type: Type of backend ('openai', 'huggingface', 'local')
            model_name: Name of the embedding model to use
            api_key: API key for OpenAI-compatible backends
            base_url: Base URL for custom OpenAI-compatible APIs
        """
        self.backend_type = backend_type.lower()
        
        try:
            if self.backend_type in ["openai", "api"]:
                if not api_key:
                    raise ValueError("api_key is required for OpenAI-compatible backends")
                self.backend = OpenAIEmbeddingBackend(api_key, base_url, model_name)
            elif self.backend_type in ["huggingface", "local"]:
                self.backend = HuggingFaceEmbeddingBackend(model_name)
            else:
                raise ValueError(f"Unsupported backend type: {backend_type}")
            
            logger.info(f"Initialized Universal Embedding adapter with backend: {backend_type}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding backend: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Convert text to embedding vector.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            return self.backend.embed_text(text)
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise

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

        try:
            return self.backend.embed_batch(texts)
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise
