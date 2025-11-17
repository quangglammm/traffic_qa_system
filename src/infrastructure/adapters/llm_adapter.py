from typing import Dict, Any, List, Optional
import json
import logging
from src.application.interfaces.i_llm_service import ILLMService
from src.application.interfaces.i_llm_backend import BaseLLMBackend
from src.domain.models import ParsedQuery

logger = logging.getLogger(__name__)


class OpenAICompatibleBackend(BaseLLMBackend):
    """Backend for OpenAI-compatible APIs (OpenAI, vLLM, etc.)"""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: str = "Qwen3-32B",
    ):
        """
        Initialize OpenAI-compatible backend.

        Args:
            api_key: API key for authentication
            base_url: Base URL for API (None for official OpenAI, custom for vLLM/others)
            model_name: Model name to use
        """
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model_name = model_name
            logger.info(
                f"Initialized OpenAI-compatible backend with model: {model_name}"
            )
        except ImportError:
            logger.error(
                "openai package not installed. Install with: pip install openai"
            )
            raise

    def generate_text(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using OpenAI-compatible API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with OpenAI-compatible API: {e}")
            return ""


class HuggingFaceBackend(BaseLLMBackend):
    """Backend for local HuggingFace models"""

    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize HuggingFace backend.

        Args:
            model_name: HuggingFace model name or path
            device: Device to run model on ('cpu' or 'cuda')
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            self.device = device
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device if device == "cuda" else None,
            )
            if device == "cpu":
                self.model = self.model.to(device)

            logger.info(f"Initialized HuggingFace backend with model: {model_name}")
        except ImportError:
            logger.error(
                "transformers not installed. Install with: pip install transformers torch"
            )
            raise

    def generate_text(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using HuggingFace model"""
        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            if prompt in response:
                response = response.replace(prompt, "")
            return response
        except Exception as e:
            logger.error(f"Error generating text with HuggingFace: {e}")
            return ""


class GeneralLLMAdapter(ILLMService):
    """General LLM Adapter supporting multiple backends"""

    def __init__(
        self,
        backend_type: str,
        model_name: str,
        device: str = "cpu",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize General LLM Adapter.

        Args:
            backend_type: Type of backend ('openai', 'huggingface', 'gemma')
            model_name: Name of the model to use
            device: Device for local models ('cpu' or 'cuda')
            api_key: API key for OpenAI-compatible backends
            base_url: Base URL for custom OpenAI-compatible APIs
        """
        self.backend_type = backend_type.lower()

        try:
            if self.backend_type in ["openai", "vllm", "api"]:
                if not api_key:
                    raise ValueError(
                        "api_key is required for OpenAI-compatible backends"
                    )
                self.backend = OpenAICompatibleBackend(api_key, base_url, model_name)
            elif self.backend_type in ["huggingface", "gemma", "local"]:
                self.backend = HuggingFaceBackend(model_name, device)
            else:
                raise ValueError(f"Unsupported backend type: {backend_type}")

            logger.info(f"Initialized General LLM adapter with backend: {backend_type}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize LLM backend: {e}. Using fallback parsing only."
            )
            self.backend = None

    def _generate_text(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using the configured backend"""
        if self.backend is None:
            return ""

        try:
            return self.backend.generate_text(prompt, max_length)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""

    def parse_query(self, user_query: str) -> ParsedQuery:
        """
        Parse user query to extract intent and entities using LLM.

        Args:
            user_query: Natural language question from user

        Returns:
            ParsedQuery object with intent, action, vehicle_type, location, etc.
        """
        # Create a structured prompt for query analysis
        parse_prompt = f"""Phân tích câu hỏi sau và trả về JSON với định dạng:
{{
    "intent": "find_penalty" hoặc "find_legal_basis" hoặc "find_supplementary",
    "action": "mô tả hành vi vi phạm",
    "vehicle_type": "ô tô" hoặc "xe máy" hoặc null,
    "location": "nội thành" hoặc "ngoại thành" hoặc null,
    "original_query": "câu hỏi gốc"
}}

Câu hỏi: {user_query}

JSON:"""

        try:
            response = self._generate_text(parse_prompt, max_length=256)

            logger.info(f"Response: {response}")

            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
            else:
                # Fallback: simple heuristic-based parsing
                parsed = self._fallback_parse(user_query)

            return ParsedQuery(
                intent=parsed.get("intent", "find_penalty"),
                action=parsed.get("action", ""),
                vehicle_type=parsed.get("vehicle_type"),
                location=parsed.get("location"),
                original_query=user_query,
            )
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            # Fallback parsing
            return self._fallback_parse_query(user_query)

    def _fallback_parse(self, user_query: str) -> dict:
        """Fallback parsing using simple heuristics"""
        query_lower = user_query.lower()

        # Determine intent
        intent = "find_penalty"
        if "điều luật" in query_lower or "căn cứ" in query_lower:
            intent = "find_legal_basis"
        elif "hình phạt bổ sung" in query_lower or "tước bằng" in query_lower:
            intent = "find_supplementary"

        # Extract vehicle type
        vehicle_type = None
        if "ô tô" in query_lower or "oto" in query_lower or "xe 4 bánh" in query_lower:
            vehicle_type = "ô tô"
        elif (
            "xe máy" in query_lower or "xe may" in query_lower or "mô tô" in query_lower
        ):
            vehicle_type = "xe máy"

        # Extract location
        location = None
        if (
            "hà nội" in query_lower
            or "hanoi" in query_lower
            or "nội thành" in query_lower
        ):
            location = "nội thành"
        elif "ngoại thành" in query_lower:
            location = "ngoại thành"

        # Extract action (simple heuristic: look for traffic violation keywords)
        action = ""
        violation_keywords = [
            "vượt đèn đỏ",
            "vượt đèn vàng",
            "vượt đèn",
            "quá tốc độ",
            "vượt quá tốc độ",
            "không đội mũ",
            "không đội nón",
            "dừng đỗ",
            "đỗ xe",
        ]
        for keyword in violation_keywords:
            if keyword in query_lower:
                action = keyword
                break

        if not action:
            # Try to extract action from the question
            action = user_query

        return {
            "intent": intent,
            "action": action,
            "vehicle_type": vehicle_type,
            "location": location,
            "original_query": user_query,
        }

    def _fallback_parse_query(self, user_query: str) -> ParsedQuery:
        """Create ParsedQuery from fallback parsing"""
        parsed = self._fallback_parse(user_query)
        return ParsedQuery(
            intent=parsed["intent"],
            action=parsed["action"],
            vehicle_type=parsed["vehicle_type"],
            location=parsed["location"],
            original_query=user_query,
        )

    def generate_response(
        self,
        violation_details: Dict[str, Any],
        parsed_query: ParsedQuery
    ) -> str:
        """
        Generate a natural, polished Vietnamese answer using the confirmed violation details.
        
        This is the FINAL user-facing response — clean, no technical jargon, no IDs.
        """
        user_query = parsed_query.original_query or "Hành vi vi phạm giao thông"

        # Extract clean data
        v = violation_details.get("violation", {})
        action = violation_details.get("action", "hành vi vi phạm").strip()
        description = v.get("detailed_description", "").strip()
        vehicle_type = v.get("vehicle_type", "")
        vehicle_str = f" đối với {vehicle_type}" if vehicle_type else ""

        # Fine
        penalty = violation_details.get("penalty") or {}
        fine_min = penalty.get("fine_min_vnd")
        fine_max = penalty.get("fine_max_vnd")
        fine_str = ""
        if fine_min is not None and fine_max is not None:
            fine_str = f"từ {fine_min:,} đến {fine_max:,} đồng".replace(",", ".")
        elif fine_min is not None:
            fine_str = f"{fine_min:,} đồng".replace(",", ".")

        # Legal basis
        law = violation_details.get("law_reference") or {}
        law_parts = []
        if law.get("point"): law_parts.append(f"Điểm {law['point']}")
        if law.get("clause"): law_parts.append(f"Khoản {law['clause']}")
        if law.get("article"): law_parts.append(f"Điều {law['article']}")
        decree = law.get("decree", "168/2024/NĐ-CP")
        law_parts.append(f"Nghị định {decree}")
        law_citation = ", ".join(law_parts)

        # Supplementary penalty
        supp_text = ""
        add_penalties = violation_details.get("additional_penalties", [])
        if add_penalties:
            main = add_penalties[0]
            desc = main.get("description", "").strip()
            if desc:
                supp_text = f"\n\nNgoài phạt tiền, bạn còn có thể bị: {desc}"

        prompt = f'''Người dùng hỏi: "{user_query}"

    Thông tin vi phạm đã xác định:
    - Hành vi: {action}
    - Mô tả chi tiết: {description}
    - Đối tượng: {vehicle_type or "mọi phương tiện"}
    - Mức phạt tiền: {fine_str or "theo quy định"}
    - Căn cứ pháp lý: {law_citation}
    - Hình phạt bổ sung: {"có" if supp_text else "không có"}

    Yêu cầu:
    - Trả lời tự nhiên, rõ ràng, lịch sự bằng tiếng Việt.
    - Không nhắc đến ID, số thứ tự, độ tương đồng, hay hệ thống.
    - Không nói "theo hệ thống", "theo dữ liệu", "tôi tìm thấy" — hãy nói như chuyên gia giao thông.
    - Nếu có hình phạt bổ sung → nhắc rõ ràng.
    - Dùng giọng điệu trung lập, chuyên nghiệp.

    Trả lời trực tiếp:'''

        answer = self._generate_text(prompt, max_length=512).strip()

        # Final fallback if LLM fails
        if not answer or len(answer) < 20:
            answer = f"Hành vi {action}{vehicle_str} sẽ bị phạt tiền {fine_str} theo {law_citation}.{supp_text or ''}"

        return answer.strip()

    def select_best_violation(
        self,
        user_query: str,
        parsed_query: ParsedQuery,
        top_violations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze top-k violations and return ONLY the best violation ID.

        Returns:
            {
                "best_violation_id": "D168-A6-C1-Pb" | None,
                "confidence": "high" | "medium" | "low",
                "reason": "Brief internal reason (for logging only)"
            }
        """
        if not top_violations:
            return {
                "best_violation_id": None,
                "confidence": "low",
                "reason": "No candidates provided",
            }

        # Build compact, structured context
        candidates = []
        for i, v in enumerate(top_violations, 1):
            viol = v.get("violation", {})
            candidates.append(
                {
                    "rank": i,
                    "id": viol.get("id", "UNKNOWN"),
                    "action": v.get("action", ""),
                    "description": viol.get("detailed_description", "")[:200],
                    "vehicle_type": viol.get("vehicle_type"),
                    "fine_range": f"{v.get('penalty', {}).get('fine_min_vnd', 0):,} - {v.get('penalty', {}).get('fine_max_vnd', 0):,} đồng",
                    "law": " → ".join(
                        filter(
                            None,
                            [
                                v.get("law_reference", {}).get("point"),
                                v.get("law_reference", {}).get("clause"),
                                v.get("law_reference", {}).get("article"),
                                v.get("law_reference", {}).get("decree"),
                            ],
                        )
                    ),
                }
            )

        candidates_str = "\n".join(
            [
                f"[{c['rank']}] {c['id']} | {c['action']} | {c['vehicle_type'] or 'Tất cả'} | {c['fine_range']} | {c['law']}"
                for c in candidates
            ]
        )

        prompt = f"""Người dùng hỏi: "{user_query}"

Danh sách các vi phạm gần nhất:
{candidates_str}

Hãy chọn ID chính xác nhất phù hợp với hành vi người dùng mô tả.

Trả về đúng định dạng JSON sau (không thêm gì khác):

{{
    "best_violation_id": "D168-A6-C1-Pb",
    "confidence": "high|medium|low"
}}

Chỉ chọn từ các ID có sẵn. Nếu không chắc chắn → chọn cái gần nhất và để confidence = "medium"."""

        raw = self._generate_text(prompt, max_length=256)

        import re, json

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                vid = result.get("best_violation_id")
                if vid and any(c["id"] == vid for c in candidates):
                    return {
                        "best_violation_id": vid,
                        "confidence": result.get("confidence", "medium"),
                        "reason": "LLM selected",
                    }
            except:
                pass

        # Final fallback
        best_guess = candidates[0]["id"]
        return {
            "best_violation_id": best_guess,
            "confidence": "low",
            "reason": "LLM failed → fallback to top-1",
        }
