from typing import Dict, Any
import json
import logging
from src.application.interfaces.i_llm_service import ILLMService
from src.domain.models import ParsedQuery

logger = logging.getLogger(__name__)

class GemmaLLMAdapter(ILLMService):
    """LLM Adapter using Gemma model for query parsing and response generation"""

    def __init__(self, model_name: str = "google/gemma-2b-it", device: str = "cpu"):
        """
        Initialize Gemma LLM Adapter.

        Args:
            model_name: Name of the Gemma model to use
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
                device_map=device if device == "cuda" else None
            )
            if device == "cpu":
                self.model = self.model.to(device)

            logger.info(f"Initialized Gemma LLM adapter with model: {model_name}")
        except ImportError:
            logger.warning("transformers not installed, using fallback JSON parsing")
            self.model = None
            self.tokenizer = None

    def _generate_text(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using the model"""
        if self.model is None or self.tokenizer is None:
            # Fallback: return empty string if model not available
            return ""

        try:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            return response
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

            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
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
                original_query=user_query
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
        elif "xe máy" in query_lower or "xe may" in query_lower or "mô tô" in query_lower:
            vehicle_type = "xe máy"

        # Extract location
        location = None
        if "hà nội" in query_lower or "hanoi" in query_lower or "nội thành" in query_lower:
            location = "nội thành"
        elif "ngoại thành" in query_lower:
            location = "ngoại thành"

        # Extract action (simple heuristic: look for traffic violation keywords)
        action = ""
        violation_keywords = [
            "vượt đèn đỏ", "vượt đèn vàng", "vượt đèn",
            "quá tốc độ", "vượt quá tốc độ",
            "không đội mũ", "không đội nón",
            "dừng đỗ", "đỗ xe"
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
            "original_query": user_query
        }

    def _fallback_parse_query(self, user_query: str) -> ParsedQuery:
        """Create ParsedQuery from fallback parsing"""
        parsed = self._fallback_parse(user_query)
        return ParsedQuery(
            intent=parsed["intent"],
            action=parsed["action"],
            vehicle_type=parsed["vehicle_type"],
            location=parsed["location"],
            original_query=user_query
        )

    def generate_response(self, violation_details: Dict[str, Any], parsed_query: ParsedQuery) -> str:
        """
        Generate natural language response from violation details.

        Args:
            violation_details: Dictionary containing violation, fine, legal_basis, supplementary
            parsed_query: The parsed query information

        Returns:
            Natural language answer string with citations
        """
        if not violation_details or not violation_details.get("violation"):
            return "Không tìm thấy dữ liệu cho vi phạm này."

        violation = violation_details.get("violation", {})
        fine = violation_details.get("fine", {})
        legal_basis = violation_details.get("legal_basis", {})
        supplementary = violation_details.get("supplementary", {})

        # Build response based on intent
        response_parts = []

        # Main answer
        if parsed_query.intent == "find_penalty":
            if fine and fine.get("min_amount") and fine.get("max_amount"):
                min_fine = int(fine["min_amount"])
                max_fine = int(fine["max_amount"])
                response_parts.append(
                    f"Hành vi {violation.get('description', '')} "
                    f"({violation.get('action', '')}) sẽ bị phạt tiền từ "
                    f"{min_fine:,} đến {max_fine:,} VNĐ."
                )
            else:
                response_parts.append(
                    f"Hành vi {violation.get('description', '')} "
                    f"({violation.get('action', '')}) sẽ bị phạt tiền."
                )

        elif parsed_query.intent == "find_legal_basis":
            response_parts.append(
                f"Hành vi {violation.get('description', '')} "
                f"({violation.get('action', '')}) được quy định tại:"
            )

        elif parsed_query.intent == "find_supplementary":
            if supplementary and supplementary.get("description"):
                response_parts.append(
                    f"Hành vi {violation.get('description', '')} "
                    f"({violation.get('action', '')}) có hình phạt bổ sung:"
                )
            else:
                response_parts.append(
                    f"Hành vi {violation.get('description', '')} "
                    f"({violation.get('action', '')}) không có hình phạt bổ sung."
                )

        # Add legal basis citation
        if legal_basis:
            legal_parts = []
            if legal_basis.get("point"):
                legal_parts.append(f"Điểm {legal_basis['point']}")
            if legal_basis.get("clause"):
                legal_parts.append(f"Khoản {legal_basis['clause']}")
            if legal_basis.get("article"):
                legal_parts.append(f"Điều {legal_basis['article']}")
            if legal_basis.get("decree"):
                legal_parts.append(f"Nghị định {legal_basis['decree']}")

            if legal_parts:
                response_parts.append(f"\nCăn cứ pháp lý: {', '.join(legal_parts)}.")

        # Add supplementary penalty
        if supplementary and supplementary.get("description"):
            if parsed_query.intent != "find_supplementary":
                response_parts.append(f"\nHình phạt bổ sung: {supplementary['description']}.")
            else:
                response_parts.append(f" {supplementary['description']}.")

        # Add fine info if not main intent
        if parsed_query.intent != "find_penalty" and fine and fine.get("min_amount") and fine.get("max_amount"):
            min_fine = int(fine["min_amount"])
            max_fine = int(fine["max_amount"])
            response_parts.append(
                f"\nMức phạt tiền: từ {min_fine:,} đến {max_fine:,} VNĐ."
            )

        return "\n".join(response_parts) if response_parts else "Không có thông tin để trả lời."

