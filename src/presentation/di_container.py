from dependency_injector import containers, providers
from src.application.use_cases.ask_question_use_case import AskQuestionUseCase
from src.infrastructure.adapters.neo4j_kg_adapter import Neo4jKGAdapter
from src.infrastructure.adapters.chroma_vs_adapter import ChromaVSAdapter
from src.infrastructure.adapters.gemma_llm_adapter import GemmaLLMAdapter
from src.infrastructure.adapters.hf_embedding_adapter import HFEmbeddingAdapter
from src.infrastructure.config import settings # File config load .env

# Đây là nơi "buộc" interface với implementation
class Container(containers.DeclarativeContainer):
    
    # 1. Cấu hình các Adapters (Singleton = chỉ tạo 1 lần)
    config = providers.Configuration()
    config.from_dict(settings.dict())

    kg_adapter = providers.Singleton(
        Neo4jKGAdapter,
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASS
    )
    
    vs_adapter = providers.Singleton(ChromaVSAdapter) # (Giả sử Chroma chạy local)
    
    llm_adapter = providers.Singleton(GemmaLLMAdapter) # (Giả sử Gemma chạy local)
    
    embed_adapter = providers.Singleton(HFEmbeddingAdapter, model_name="bkai-model...")

    # 2. Cấu hình Use Cases (Factory = tạo mới mỗi lần gọi)
    # Tự động "tiêm" các adapter ở trên vào
    ask_question_use_case = providers.Factory(
        AskQuestionUseCase,
        llm_service=llm_adapter,
        vector_store=vs_adapter,
        kg=kg_adapter,
        embedding_service=embed_adapter
    )
    
    # (Tương tự cho LoadDataUseCase)