from pydantic import BaseModel
from typing import Optional, List

# Dùng Pydantic (hoặc dataclasses) để định nghĩa cấu trúc dữ liệu
class Fine(BaseModel):
    min_amount: int
    max_amount: int

class SupplementaryPenalty(BaseModel):
    description: str

class LegalBasis(BaseModel):
    decree: str  # Nghị định
    article: str # Điều
    clause: str  # Khoản
    point: str   # Điểm

class Violation(BaseModel):
    id: str
    description: str # Mô tả theo luật
    vehicle_type: str

class QueryResponse(BaseModel):
    answer: str
    violation_found: Optional[Violation] = None
    fine: Optional[Fine] = None
    legal_basis: Optional[LegalBasis] = None
    supplementary: Optional[SupplementaryPenalty] = None
    citation: str # Trích dẫn