from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Neo4j Configuration
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASS: str = "password"

    # API Configuration
    API_KEY: Optional[str] = None
    BASE_URL: Optional[str] = None

    # LLM Configuration
    LLM_BACKEND_TYPE: str = "huggingface"  # "openai" or "huggingface"
    LLM_MODEL_NAME: str = ""
    LLM_DEVICE: str = "cpu"  # "cpu" or "cuda"

    # Embedding Configuration
    EMBEDDING_BACKEND_TYPE: str = "openai"  # "openai" or "huggingface"
    EMBEDDING_MODEL_NAME: str = "Vienamese_Embedding"

    # Vector Store Configuration
    CHROMA_COLLECTION_NAME: str = "traffic_violations"
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"

    # Similarity Threshold
    MIN_SIMILARITY_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create global settings instance
settings = Settings()

