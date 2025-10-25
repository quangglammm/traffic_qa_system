from src.domain.models import QueryResponse
from src.application.interfaces.i_llm_service import ILLMService
from src.application.interfaces.i_vector_store import IVectorStore
from src.application.interfaces.i_knowledge_graph import IKnowledgeGraph
from src.application.interfaces.i_embedding_service import IEmbeddingService

class AskQuestionUseCase:
    # Use case này "phụ thuộc" vào các interface, không phải implementation
    def __init__(self, 
                 llm_service: ILLMService, 
                 vector_store: IVectorStore, 
                 kg: IKnowledgeGraph,
                 embedding_service: IEmbeddingService):
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.kg = kg
        self.embedding_service = embedding_service

    def execute(self, user_query: str) -> QueryResponse:
        # 1. Dùng LLM để hiểu câu hỏi
        parsed_query = self.llm_service.parse_query(user_query)
        # parsed_query = {"action": "vượt đèn đỏ", "vehicle": "ô tô"}

        # 2. Dùng Embedding Model để vector hóa hành vi
        query_vector = self.embedding_service.embed_text(parsed_query["action"])

        # 3. Dùng Vector Store để tìm K vi phạm gần nhất
        # (Giả sử Vector Store chỉ lưu ID và vector của hành vi)
        similar_ids = self.vector_store.search_similar(query_vector, k=3)
        # similar_ids = ["Loi_002", "Loi_001", "Loi_105"]

        # 4. (TODO) Lọc và Xếp hạng (phần logic phức tạp của bạn)
        # Ví dụ: Dùng KG để lọc lại theo "vehicle"
        # ...
        best_violation_id = "Loi_002" # Giả sử tìm được ID tốt nhất

        # 5. Dùng Knowledge Graph để lấy chi tiết, đáng tin cậy
        if not best_violation_id:
            return QueryResponse(answer="Không tìm thấy dữ liệu cho vi phạm này.")
            
        details = self.kg.get_violation_details(best_violation_id)
        # details = { "violation": {...}, "fine": {...}, "legal_basis": {...} }

        # 6. (TODO) Tổng hợp câu trả lời
        # ...
        
        return QueryResponse(
            answer=f"Hành vi {details['violation'].description}...",
            # ... điền các trường còn lại
            citation=f"{details['legal_basis'].article}, {details['legal_basis'].decree}"
        )