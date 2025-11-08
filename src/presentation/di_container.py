"""
Dependency Injection Container for the Traffic QA System.

This module defines the dependency injection container that binds interfaces
to their implementations and manages the lifecycle of application components.
"""
from dependency_injector import containers, providers
from src.application.use_cases.ask_question_use_case import AskQuestionUseCase
from src.infrastructure.adapters.neo4j_kg_adapter import Neo4jKGAdapter
from src.infrastructure.adapters.chroma_vs_adapter import ChromaVSAdapter
from src.infrastructure.adapters.llm_adapter import GeneralLLMAdapter
from src.infrastructure.adapters.embedding_adapter import EmbeddingAdapter
from src.infrastructure.config import settings


class Container(containers.DeclarativeContainer):
    """
    Dependency injection container that wires interfaces to implementations.

    This container manages:
    - Infrastructure adapters (singletons for performance)
    - Use cases (factories for stateless operations)
    - Configuration (loaded from environment variables)
    """

    # Configuration provider - loads settings from environment
    config = providers.Configuration()
    config.from_dict(settings.model_dump())
    print("Loaded configuration:", settings.model_dump())

    # Infrastructure adapters (singletons - created once, reused across requests)
    kg_adapter = providers.Singleton(
        Neo4jKGAdapter,
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASS
    )

    vs_adapter = providers.Singleton(
        ChromaVSAdapter,
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY
    )

    llm_adapter = providers.Singleton(
        GeneralLLMAdapter,
        backend_type=config.LLM_BACKEND_TYPE,
        model_name=config.LLM_MODEL_NAME,
        device=config.LLM_DEVICE,
        api_key=config.API_KEY,
        base_url=config.BASE_URL
    )

    embed_adapter = providers.Singleton(
        EmbeddingAdapter,
        backend_type=config.EMBEDDING_BACKEND_TYPE,
        model_name=config.EMBEDDING_MODEL_NAME,
        api_key=config.API_KEY,
        base_url=config.BASE_URL
    )

    # Use cases (factories - new instance per request for stateless operations)
    # Dependencies are automatically injected from the adapters above
    ask_question_use_case = providers.Factory(
        AskQuestionUseCase,
        llm_service=llm_adapter,
        vector_store=vs_adapter,
        kg=kg_adapter,
        embedding_service=embed_adapter
    )
