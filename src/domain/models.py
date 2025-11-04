from pydantic import BaseModel
from typing import Optional, List

# Dùng Pydantic (hoặc dataclasses) để định nghĩa cấu trúc dữ liệu
class Fine(BaseModel):
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None

class SupplementaryPenalty(BaseModel):
    description: Optional[str] = None

class LegalBasis(BaseModel):
    decree: Optional[str] = None  # Nghị định
    article: Optional[str] = None # Điều
    clause: Optional[str] = None  # Khoản
    point: Optional[str] = None   # Điểm

class Violation(BaseModel):
    id: str
    description: str # Mô tả theo luật
    vehicle_type: Optional[str] = None
    action: Optional[str] = None

class ParsedQuery(BaseModel):
    """Result of query analysis - intent classification and entity extraction"""
    intent: str  # "find_penalty", "find_legal_basis", "find_supplementary"
    action: str  # e.g., "vượt đèn vàng"
    vehicle_type: Optional[str] = None  # e.g., "ô tô", "xe máy"
    location: Optional[str] = None  # e.g., "Hà Nội" -> "nội thành"
    original_query: str

class QueryResponse(BaseModel):
    answer: str
    violation_found: Optional[Violation] = None
    fine: Optional[Fine] = None
    legal_basis: Optional[LegalBasis] = None
    supplementary: Optional[SupplementaryPenalty] = None
    citation: str = "" # Trích dẫn