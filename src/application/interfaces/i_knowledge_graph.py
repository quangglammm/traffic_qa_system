from abc import ABC, abstractmethod
from typing import List
from src.domain.models import Violation, Fine, LegalBasis # Phụ thuộc vào Domain

# Đây là một "Hợp đồng" (Port)
class IKnowledgeGraph(ABC):

    @abstractmethod
    def add_violation_batch(self, violations_data: List[dict]):
        # Thêm hàng loạt vi phạm từ file JSON
        pass

    @abstractmethod
    def get_violation_details(self, violation_id: str) -> dict:
        # Lấy chi tiết (phạt, luật,...) từ ID vi phạm
        pass