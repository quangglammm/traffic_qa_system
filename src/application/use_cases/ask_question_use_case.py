from src.domain.models import QueryResponse, Violation, Fine, LegalBasis, SupplementaryPenalty
from src.application.interfaces.i_llm_service import ILLMService
from src.application.interfaces.i_vector_store import IVectorStore
from src.application.interfaces.i_knowledge_graph import IKnowledgeGraph
from src.application.interfaces.i_embedding_service import IEmbeddingService
import logging

logger = logging.getLogger(__name__)

class AskQuestionUseCase:
    """
    Use case implementing the advanced 5-step question answering flow:
    1. Query Analysis (LLM) - Intent classification and entity extraction
    2. Semantic Search (Vector) - Find similar violations using embeddings
    3. Filtering & Ranking (KG) - Filter by vehicle type and other entities
    4. Knowledge Retrieval (KG) - Get detailed violation information
    5. Response Generation (LLM) - Generate natural language answer with citations
    """

    def __init__(self,
                 llm_service: ILLMService,
                 vector_store: IVectorStore,
                 kg: IKnowledgeGraph,
                 embedding_service: IEmbeddingService):
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.kg = kg
        self.embedding_service = embedding_service
        # Threshold can be configured via config, default is 0.7
        try:
            from src.infrastructure.config import settings
            self.min_similarity_threshold = settings.MIN_SIMILARITY_THRESHOLD
        except:
            self.min_similarity_threshold = 0.7  # Default threshold

    def execute(self, user_query: str) -> QueryResponse:
        """
        Execute the complete question answering flow.

        Args:
            user_query: Natural language question from user

        Returns:
            QueryResponse with answer and citation
        """
        try:
            # ===== BƯỚC 1: PHÂN TÍCH TRUY VẤN (Query Analysis) - Dùng LLM =====
            logger.info("Step 1: Query Analysis - Parsing user query")
            parsed_query = self.llm_service.parse_query(user_query)
            logger.info(f"Parsed query: intent={parsed_query.intent}, action={parsed_query.action}, "
                       f"vehicle_type={parsed_query.vehicle_type}, location={parsed_query.location}")

            if not parsed_query.action:
                logger.warning("No action extracted from query")
                return QueryResponse(
                    answer="Không thể xác định hành vi vi phạm từ câu hỏi. Vui lòng cung cấp thêm thông tin.",
                    citation=""
                )

            # ===== BƯỚC 2: TÌM KIẾM NGỮ NGHĨA (Semantic Search) - Dùng Vector =====
            logger.info("Step 2: Semantic Search - Finding similar violations")

            # Create embedding for the action or full query
            query_text = parsed_query.action if parsed_query.action else user_query
            query_vector = self.embedding_service.embed_text(query_text)

            # Search for k-NN violations (k=5)
            similar_results = self.vector_store.search_similar(
                query_vector,
                k=5,
                min_similarity=self.min_similarity_threshold
            )

            if not similar_results:
                logger.warning(f"No similar violations found above threshold {self.min_similarity_threshold}")
                return QueryResponse(
                    answer="Không có dữ liệu / Không tìm thấy vi phạm tương ứng.",
                    citation=""
                )

            # Extract violation IDs from search results
            similar_ids = [vid for vid, score in similar_results]
            similarity_scores = {vid: score for vid, score in similar_results}

            logger.info(f"Found {len(similar_ids)} similar violations: {similar_ids}")

            # Check if highest similarity is below threshold
            highest_similarity = similarity_scores[similar_ids[0]] if similar_ids else 0.0
            if highest_similarity < self.min_similarity_threshold:
                logger.warning(f"Highest similarity {highest_similarity} below threshold {self.min_similarity_threshold}")
                return QueryResponse(
                    answer="Không có dữ liệu / Không tìm thấy vi phạm tương ứng.",
                    citation=""
                )

            # ===== BƯỚC 3: LỌC VÀ XẾP HẠNG (Filtering & Ranking) - Dùng KG =====
            logger.info("Step 3: Filtering & Ranking - Filtering by vehicle type")

            # Filter violations by vehicle type if specified
            filtered_ids = self.kg.filter_violations_by_vehicle_type(
                similar_ids,
                parsed_query.vehicle_type
            )

            # If filtering removed all results, use original results
            if not filtered_ids:
                logger.warning("Filtering removed all results, using original similar violations")
                filtered_ids = similar_ids
            else:
                logger.info(f"Filtered to {len(filtered_ids)} violations: {filtered_ids}")

            # Select best violation (highest similarity from filtered results)
            best_violation_id = None
            best_similarity = 0.0

            for vid in filtered_ids:
                if vid in similarity_scores and similarity_scores[vid] > best_similarity:
                    best_similarity = similarity_scores[vid]
                    best_violation_id = vid

            # Fallback to first filtered ID if no match found
            if not best_violation_id and filtered_ids:
                best_violation_id = filtered_ids[0]
                best_similarity = similarity_scores.get(best_violation_id, 0.0)

            if not best_violation_id:
                logger.warning("No best violation ID found")
                return QueryResponse(
                    answer="Không có dữ liệu / Không tìm thấy vi phạm tương ứng.",
                    citation=""
                )

            logger.info(f"Selected best violation: {best_violation_id} (similarity: {best_similarity:.3f})")

            # ===== BƯỚC 4: TRUY XUẤT TRI THỨC (Knowledge Retrieval) - Dùng KG =====
            logger.info(f"Step 4: Knowledge Retrieval - Getting details for violation {best_violation_id}")

            violation_details = self.kg.get_violation_details(best_violation_id)

            if not violation_details or not violation_details.get("violation"):
                logger.warning(f"Could not retrieve details for violation {best_violation_id}")
                return QueryResponse(
                    answer="Không có dữ liệu / Không tìm thấy vi phạm tương ứng.",
                    citation=""
                )

            # ===== BƯỚC 5: TẠO CÂU TRẢ LỜI (Response Generation) =====
            logger.info("Step 5: Response Generation - Generating natural language answer")

            # Generate natural language response using LLM
            answer = self.llm_service.generate_response(violation_details, parsed_query)

            # Build citation from legal basis
            citation = ""
            legal_basis = violation_details.get("legal_basis")
            if legal_basis:
                citation_parts = []
                if legal_basis.get("point"):
                    citation_parts.append(f"Điểm {legal_basis['point']}")
                if legal_basis.get("clause"):
                    citation_parts.append(f"Khoản {legal_basis['clause']}")
                if legal_basis.get("article"):
                    citation_parts.append(f"Điều {legal_basis['article']}")
                if legal_basis.get("decree"):
                    citation_parts.append(f"Nghị định {legal_basis['decree']}")

                if citation_parts:
                    citation = ", ".join(citation_parts)

            # Build domain models for response
            violation_data = violation_details.get("violation", {})
            violation = Violation(
                id=violation_data.get("id", ""),
                description=violation_data.get("description", ""),
                vehicle_type=violation_data.get("vehicle_type"),
                action=violation_data.get("action")
            ) if violation_data else None

            fine_data = violation_details.get("fine", {})
            fine = Fine(
                min_amount=fine_data.get("min_amount"),
                max_amount=fine_data.get("max_amount")
            ) if fine_data else None

            legal_basis_model = LegalBasis(
                decree=legal_basis.get("decree") if legal_basis else None,
                article=legal_basis.get("article") if legal_basis else None,
                clause=legal_basis.get("clause") if legal_basis else None,
                point=legal_basis.get("point") if legal_basis else None
            ) if legal_basis else None

            supplementary_data = violation_details.get("supplementary", {})
            supplementary = SupplementaryPenalty(
                description=supplementary_data.get("description")
            ) if supplementary_data and supplementary_data.get("description") else None

            logger.info("Successfully generated response")

            return QueryResponse(
                answer=answer,
                violation_found=violation,
                fine=fine,
                legal_basis=legal_basis_model,
                supplementary=supplementary,
                citation=citation
            )

        except Exception as e:
            logger.error(f"Error in ask question use case: {e}", exc_info=True)
            return QueryResponse(
                answer=f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}",
                citation=""
            )
