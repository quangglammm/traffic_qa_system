from neo4j import GraphDatabase
from typing import List, Optional
from src.application.interfaces.i_knowledge_graph import IKnowledgeGraph
import logging

logger = logging.getLogger(__name__)

class Neo4jKGAdapter(IKnowledgeGraph):

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # Implement function
    def add_violation_batch(self, violations_data: List[dict]):
        """
        Add violations in batch to Neo4j knowledge graph.

        Args:
            violations_data: List of violation dictionaries from traffic.json

        Raises:
            ValueError: If violations_data is empty or invalid
            Exception: If database operation fails
        """
        if not violations_data:
            raise ValueError("violations_data cannot be empty")

        if not isinstance(violations_data, list):
            raise ValueError("violations_data must be a list")

        try:
            # Convert data from /data/traffic.json
            def parse_legal_clause(dieu_luat_value: str):
                if not dieu_luat_value:
                    return {"article": None, "clause": None, "point": None}
                parts = [p.strip() for p in str(dieu_luat_value).split(",")]
                article = parts[0] if len(parts) > 0 else None
                clause = parts[1] if len(parts) > 1 else None
                point = parts[2] if len(parts) > 2 else None
                return {"article": article, "clause": clause, "point": point}

            transformed = []
            for idx, item in enumerate(violations_data):
                try:
                    if not isinstance(item, dict):
                        logger.warning(f"Skipping item at index {idx}: not a dictionary")
                        continue

                    if not item.get("id"):
                        logger.warning(f"Skipping item at index {idx}: missing 'id' field")
                        continue

                    legal = parse_legal_clause(item.get("dieu_luat"))

                    # Validate and convert fine amounts
                    fine_min = item.get("phat_min")
                    fine_max = item.get("phat_max")
                    try:
                        fine_min = float(fine_min) if fine_min is not None else None
                        fine_max = float(fine_max) if fine_max is not None else None
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid fine amount for violation {item.get('id')}: {e}")
                        fine_min = None
                        fine_max = None

                    transformed.append({
                        "id": item.get("id"),
                        "action": item.get("hanh_vi_chung"),
                        "description": item.get("mo_ta_luat"),
                        "vehicle_type": item.get("loai_xe"),
                        "fine_min": fine_min,
                        "fine_max": fine_max,
                        "decree": item.get("nghi_dinh"),
                        "article": legal["article"],
                        "clause": legal["clause"],
                        "point": legal["point"],
                        "supplementary": item.get("bo_sung")
                    })
                except Exception as e:
                    logger.error(f"Error processing violation at index {idx}: {e}")
                    continue

            if not transformed:
                raise ValueError("No valid violations to import after processing")

            cypher = """
            UNWIND $violations AS v
            MERGE (viol:Violation {id: v.id})
              ON CREATE SET viol.description = v.description,
                            viol.vehicle_type = v.vehicle_type,
                            viol.action = v.action
              ON MATCH SET  viol.description = v.description,
                            viol.vehicle_type = v.vehicle_type,
                            viol.action = v.action

            // Fine
            MERGE (fine:Fine {violation_id: v.id})
              ON CREATE SET fine.min_amount = v.fine_min, fine.max_amount = v.fine_max
              ON MATCH SET  fine.min_amount = v.fine_min, fine.max_amount = v.fine_max
            MERGE (viol)-[:HAS_FINE]->(fine)

            // Legal Basis
            MERGE (law:LegalBasis {decree: v.decree, article: v.article, clause: v.clause, point: v.point})
            MERGE (viol)-[:STIPULATED_IN]->(law)

            // Supplementary penalty (optional)
            FOREACH (_ IN CASE WHEN v.supplementary IS NULL THEN [] ELSE [1] END |
              MERGE (sp:SupplementaryPenalty {description: v.supplementary})
              MERGE (viol)-[:HAS_SUPPLEMENTARY]->(sp)
            )
            """

            with self.driver.session() as session:
                session.execute_write(lambda tx: tx.run(cypher, violations=transformed))

            logger.info(f"Successfully imported {len(transformed)} violations into KG.")
            print(f"Successfully imported {len(transformed)} violations into KG.")

        except Exception as e:
            logger.error(f"Error importing violations batch: {e}", exc_info=True)
            raise

    # Implement function
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
        if not violation_id:
            raise ValueError("violation_id cannot be empty")

        if not isinstance(violation_id, str):
            raise ValueError("violation_id must be a string")

        try:
            query = """
            MATCH (v:Violation {id: $id})
            OPTIONAL MATCH (v)-[:HAS_FINE]->(f:Fine)
            OPTIONAL MATCH (v)-[:STIPULATED_IN]->(l:LegalBasis)
            OPTIONAL MATCH (v)-[:HAS_SUPPLEMENTARY]->(sp:SupplementaryPenalty)
            RETURN v, f, l, sp
            """

            with self.driver.session() as session:
                result = session.run(query, id=violation_id).single()

                if result is None:
                    logger.warning(f"Violation with id '{violation_id}' not found")
                    return {
                        "violation": None,
                        "fine": None,
                        "legal_basis": None,
                        "supplementary": None
                    }

                # Extract violation node
                violation_node = result.get("v")
                violation = None
                if violation_node:
                    violation = {
                        "id": violation_node.get("id"),
                        "description": violation_node.get("description"),
                        "vehicle_type": violation_node.get("vehicle_type"),
                        "action": violation_node.get("action")
                    }

                # Extract fine node
                fine_node = result.get("f")
                fine = None
                if fine_node:
                    fine = {
                        "min_amount": fine_node.get("min_amount"),
                        "max_amount": fine_node.get("max_amount")
                    }

                # Extract legal basis node
                legal_basis_node = result.get("l")
                legal_basis = None
                if legal_basis_node:
                    legal_basis = {
                        "decree": legal_basis_node.get("decree"),
                        "article": legal_basis_node.get("article"),
                        "clause": legal_basis_node.get("clause"),
                        "point": legal_basis_node.get("point")
                    }

                # Extract supplementary penalty node
                supplementary_node = result.get("sp")
                supplementary = None
                if supplementary_node:
                    supplementary = {
                        "description": supplementary_node.get("description")
                    }

                result_dict = {
                    "violation": violation,
                    "fine": fine,
                    "legal_basis": legal_basis,
                    "supplementary": supplementary
                }

                logger.info(f"Successfully retrieved violation details for id '{violation_id}'")
                return result_dict

        except Exception as e:
            logger.error(f"Error retrieving violation details for id '{violation_id}': {e}", exc_info=True)
            raise

    def filter_violations_by_vehicle_type(self, violation_ids: List[str], vehicle_type: Optional[str]) -> List[str]:
        """
        Filter violations by vehicle type from a list of violation IDs.

        Args:
            violation_ids: List of violation IDs to filter
            vehicle_type: Vehicle type to filter by (e.g., "ô tô", "xe máy", "o_to", "xe_may")

        Returns:
            List of violation IDs that match the vehicle type filter
        """
        if not violation_ids:
            return []

        if not vehicle_type:
            # If no vehicle type specified, return all violations
            return violation_ids

        try:
            # Normalize vehicle type (handle both Vietnamese and code formats)
            vehicle_type_mapping = {
                "ô tô": "o_to",
                "oto": "o_to",
                "xe 4 bánh": "o_to",
                "xe máy": "xe_may",
                "xe may": "xe_may",
                "mô tô": "xe_may",
                "moto": "xe_may"
            }

            normalized_vehicle_type = vehicle_type.lower().strip()
            if normalized_vehicle_type in vehicle_type_mapping:
                normalized_vehicle_type = vehicle_type_mapping[normalized_vehicle_type]

            # Query to filter violations by vehicle type
            query = """
            MATCH (v:Violation)
            WHERE v.id IN $violation_ids
            AND (
                v.vehicle_type = $vehicle_type
                OR v.vehicle_type = $normalized_vehicle_type
                OR toLower(v.vehicle_type) = toLower($vehicle_type)
            )
            RETURN v.id as violation_id
            ORDER BY v.id
            """

            with self.driver.session() as session:
                result = session.run(
                    query,
                    violation_ids=violation_ids,
                    vehicle_type=vehicle_type,
                    normalized_vehicle_type=normalized_vehicle_type
                )

                filtered_ids = [record["violation_id"] for record in result]

                # If no matches found, try fuzzy matching with vehicle type codes
                if not filtered_ids:
                    # Try to match with partial vehicle type
                    fuzzy_query = """
                    MATCH (v:Violation)
                    WHERE v.id IN $violation_ids
                    AND (
                        v.vehicle_type CONTAINS $vehicle_type_part
                        OR $vehicle_type_part CONTAINS v.vehicle_type
                    )
                    RETURN v.id as violation_id
                    ORDER BY v.id
                    """

                    # Extract main part of vehicle type (e.g., "o_to" from "o_to")
                    vehicle_type_part = normalized_vehicle_type.split("_")[0] if "_" in normalized_vehicle_type else normalized_vehicle_type

                    result = session.run(
                        fuzzy_query,
                        violation_ids=violation_ids,
                        vehicle_type_part=vehicle_type_part
                    )
                    filtered_ids = [record["violation_id"] for record in result]

                logger.info(f"Filtered {len(filtered_ids)} violations from {len(violation_ids)} by vehicle_type '{vehicle_type}'")
                return filtered_ids

        except Exception as e:
            logger.error(f"Error filtering violations by vehicle type: {e}", exc_info=True)
            # Fallback: return all violations if filtering fails
            return violation_ids