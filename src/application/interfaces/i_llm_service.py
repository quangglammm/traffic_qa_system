from abc import ABC, abstractmethod
from typing import Dict, Any, List
from src.domain.models import ParsedQuery


class ILLMService(ABC):
    """Interface for LLM service to parse queries and generate responses"""

    @abstractmethod
    def parse_query(self, user_query: str) -> ParsedQuery:
        """
        Parse user query to extract intent and entities.

        Args:
            user_query: Natural language question from user

        Returns:
            ParsedQuery object with intent, action, vehicle_type, location, etc.
        """
        pass

    @abstractmethod
    def generate_response(
        self, violation_details: Dict[str, Any], parsed_query: ParsedQuery
    ) -> str:
        """
        Generate natural language response from violation details.

        Args:
            violation_details: Dictionary containing violation, fine, legal_basis, supplementary
            parsed_query: The parsed query information

        Returns:
            Natural language answer string with citations
        """
        pass

    @abstractmethod
    def select_best_violation(
        self,
        user_query: str,
        parsed_query: ParsedQuery,
        top_violations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        pass
