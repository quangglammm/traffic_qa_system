from abc import ABC, abstractmethod
from typing import List, Optional

class IKnowledgeGraph(ABC):

    @abstractmethod
    def add_violation_batch(self, violations_data: List[dict]):
        """
        Add violations in batch to Neo4j knowledge graph.

        Args:
            violations_data: List of violation dictionaries from traffic.json

        Raises:
            ValueError: If violations_data is empty or invalid
            Exception: If database operation fails
        """
        pass

    @abstractmethod
    def get_violation_details(self, violation_id: str) -> dict:
        """
        Get detailed information about a violation from the knowledge graph.

        Args:
            violation_id: The ID of the violation to retrieve

        Returns:
            Dictionary containing violation, fine, legal_basis, and supplementary penalty information

        Raises:
            ValueError: If violation_id is empty or invalid
            Exception: If database operation fails
        """
        pass

    @abstractmethod
    def filter_violations_by_vehicle_type(self, violation_ids: List[str], vehicle_type: Optional[str]) -> List[str]:
        """
        Filter violations by vehicle type from a list of violation IDs.

        Args:
            violation_ids: List of violation IDs to filter
            vehicle_type: Vehicle type to filter by (e.g., "ô tô", "xe máy", "o_to", "xe_may")

        Returns:
            List of violation IDs that match the vehicle type filter
        """
        pass