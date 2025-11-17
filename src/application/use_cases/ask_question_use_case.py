from src.domain.models import (
    QueryResponse,
    Violation,
    Fine,
    LegalBasis,
    SupplementaryPenalty,
)
from src.application.interfaces.i_llm_service import ILLMService
from src.application.interfaces.i_vector_store import IVectorStore
from src.application.interfaces.i_knowledge_graph import IKnowledgeGraph
from src.application.interfaces.i_embedding_service import IEmbeddingService
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AskQuestionUseCase:
    """
    Production-ready Top-K Reasoning QA System (Clean & User-Focused)
    No technical details about decree changes or similarity scores shown to user
    """

    def __init__(
        self,
        llm_service: ILLMService,
        vector_store: IVectorStore,
        kg: IKnowledgeGraph,
        embedding_service: IEmbeddingService,
    ):
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.kg = kg
        self.embedding_service = embedding_service

        try:
            from src.infrastructure.config import settings

            self.min_similarity = getattr(settings, "MIN_SIMILARITY_THRESHOLD", 0.72)
            self.top_k_vector = getattr(settings, "TOP_K_VECTOR", 8)
            self.top_k_final = getattr(settings, "TOP_K_FINAL", 3)
        except:
            self.min_similarity = 0.72
            self.top_k_vector = 8
            self.top_k_final = 3

    def execute(self, user_query: str) -> QueryResponse:
        try:
            # Step 1: Parse query
            logger.info("Step 1: Parsing user question")
            parsed = self.llm_service.parse_query(user_query)

            if not parsed.action:
                return QueryResponse(
                    answer="Tôi chưa hiểu rõ bạn đang hỏi về hành vi nào. Bạn có thể mô tả cụ thể hơn được không ạ?",
                    citation="",
                )

            # Step 2: Hybrid retrieval
            query_vector = self.embedding_service.embed_text(parsed.action)

            metadata_filter = {}
            if parsed.vehicle_type:
                vt = parsed.vehicle_type.lower()
                if any(x in vt for x in ["ô tô", "xe hơi", "xe con", "oto"]):
                    metadata_filter["vehicle_type"] = "ô tô"
                elif any(x in vt for x in ["xe máy", "mô tô", "xe gắn máy"]):
                    metadata_filter["vehicle_type"] = "xe máy"

            candidates = self.vector_store.search_similar(
                query_embedding=query_vector,
                k=self.top_k_vector,
                min_similarity=self.min_similarity,
                filter_metadata=metadata_filter or None,
            )

            if not candidates:
                return QueryResponse(
                    answer="Hiện tại tôi không tìm thấy quy định nào phù hợp với mô tả của bạn. "
                    "Bạn có thể thử diễn đạt lại không ạ?",
                    citation="",
                )

            # Step 3: Enrich top candidates
            enriched: List[Dict[str, Any]] = []
            for violation_id, _, _ in candidates[: self.top_k_final]:
                details = self.kg.get_violation_details(violation_id)
                if details and details.get("violation"):
                    enriched.append(details)

            if not enriched:
                return QueryResponse(
                    answer="Đã tìm thấy một số trường hợp tương tự nhưng không thể hiển thị chi tiết. "
                    "Vui lòng thử lại sau nhé!",
                    citation="",
                )

            # Step 4: LLM generates clean, natural answer from top-k
            best_violation = self.llm_service.select_best_violation(
                user_query=user_query, parsed_query=parsed, top_violations=enriched
            )

            # Step 5: Extract best violation for structured output
            best_id = self._extract_best_id(best_violation, enriched)
            best = next(
                (v for v in enriched if v["violation"]["id"] == best_id), enriched[0]
            )

            # Step 6
            answer = self.llm_service.generate_natural_answer(parsed, best)

            v = best["violation"]
            p = best.get("penalty") or {}
            law = best.get("law_reference") or {}
            add_penalties = best.get("additional_penalties", [])

            # Build clean citation
            parts = []
            if law.get("point"):
                parts.append(f"Điểm {law['point']}")
            if law.get("clause"):
                parts.append(f"Khoản {law['clause']}")
            if law.get("article"):
                parts.append(f"Điều {law['article']}")
            if law.get("decree"):
                parts.append(f"Nghị định {law['decree']}")
            citation = ", ".join(parts) if parts else ""

            # Domain models
            violation = Violation(
                id=v["id"],
                description=v.get("detailed_description", ""),
                vehicle_type=v.get("vehicle_type"),
                action=best.get("action"),
            )

            fine = (
                Fine(min_amount=p.get("fine_min_vnd"), max_amount=p.get("fine_max_vnd"))
                if p
                else None
            )

            legal_basis = (
                LegalBasis(
                    decree=law.get("decree"),
                    article=law.get("article"),
                    clause=law.get("clause"),
                    point=law.get("point"),
                )
                if law
                else None
            )

            supp = add_penalties[0] if add_penalties else {}
            supplementary = (
                SupplementaryPenalty(
                    description=supp.get("description", ""),
                    type=supp.get("type"),
                    condition=supp.get("condition"),
                )
                if supp
                else None
            )

            return QueryResponse(
                answer=answer.strip(),
                violation_found=violation,
                fine=fine,
                legal_basis=legal_basis,
                supplementary=supplementary,
                citation=citation,
            )

        except Exception as e:
            logger.error(f"Error in AskQuestionUseCase: {e}", exc_info=True)
            return QueryResponse(
                answer="Xin lỗi, hiện tại hệ thống đang gặp sự cố. Bạn vui lòng thử lại sau ít phút nhé!",
                citation="",
            )

    def _extract_best_id(self, answer: str, candidates: List[Dict]) -> str:
        text = answer.lower()
        for c in candidates:
            vid = c["violation"]["id"].lower()
            if vid in text:
                return c["violation"]["id"]
        return candidates[0]["violation"]["id"]
