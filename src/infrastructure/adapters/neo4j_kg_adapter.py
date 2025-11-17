from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable, AuthError, TransientError
from typing import List, Optional, Dict, Any
from src.application.interfaces.i_knowledge_graph import IKnowledgeGraph
import logging
import hashlib
import time

logger = logging.getLogger(__name__)
logging.getLogger("neo4j").setLevel(logging.WARNING)


class Neo4jKGAdapter(IKnowledgeGraph):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        max_connection_pool_size: int = 50,
        connection_timeout: int = 60,
    ):
        try:
            self.driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_lifetime=3600,
                max_connection_pool_size=max_connection_pool_size,
                connection_acquisition_timeout=connection_timeout,
            )
            with self.driver.session() as session:
                session.run("RETURN 1 AS test")
            logger.info(f"Successfully connected to Neo4j at {uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Neo4j connection: {e}")
            raise

    def close(self):
        if hasattr(self, "driver") and self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    # === Private Helpers ===
    def _execute_write(self, query: str, **params):
        with self.driver.session() as session:
            for attempt in range(3):
                try:
                    session.execute_write(lambda tx: tx.run(query, **params))
                    return
                except TransientError as e:
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
                    logger.warning(f"Transient error, retry {attempt + 1}: {e}")
                except Neo4jError as e:
                    logger.error(f"Neo4j error: {e}")
                    raise

    def _generate_penalty_key(self, data: Dict[str, Any]) -> str:
        parts = [
            str(data.get("type", "")).strip(),
            str(data.get("condition", "")).strip().lower(),
            str(data.get("duration_min_months") or ""),
            str(data.get("duration_max_months") or ""),
            str(data.get("minus_point") or ""),
            str(data.get("full_reference", "")).strip(),
        ]
        key = "|".join(parts)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _parse_fine(self, amount: Any) -> Optional[int]:
        if amount is None:
            return None
        try:
            if isinstance(amount, str):
                amount = amount.replace("_", "").replace(",", "").strip()
            return int(float(amount))
        except (ValueError, TypeError):
            return None

    def _build_law_hierarchy(
        self, decree: str, article: str, clause: str, point: str
    ) -> Dict[str, str]:
        decree = str(decree or "").strip()
        article = str(article or "").strip()
        clause = str(clause or "").strip()
        point = str(point or "").strip()

        return {
            "decree": decree,
            "article": article,
            "clause": clause,
            "point": point,
            "full_reference": f"{decree}|{article}|{clause}|{point}" if decree else "",
        }

    # Add this method to Neo4jKGAdapter class
    def _ensure_schema(self):
        constraints = [
            "CREATE CONSTRAINT violation_id IF NOT EXISTS FOR (v:Violation) REQUIRE v.id IS UNIQUE",
            "CREATE CONSTRAINT action_name IF NOT EXISTS FOR (a:CanonicalAction) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT legal_text IF NOT EXISTS FOR (l:LegalDescription) REQUIRE l.text IS UNIQUE",
            "CREATE CONSTRAINT penalty_key IF NOT EXISTS FOR (ap:AdditionalPenalty) REQUIRE ap.penalty_key IS UNIQUE",
            "CREATE CONSTRAINT law_ref IF NOT EXISTS FOR (lr:LawReference) REQUIRE lr.full_reference IS UNIQUE",
        ]

        with self.driver.session() as session:
            for stmt in constraints:
                try:
                    session.run(stmt)
                except Neo4jError as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Could not create constraint: {e}")

    # === Required Interface Methods ===
    def add_violation_batch(self, violations_data: List[dict]):
        if not violations_data or not isinstance(violations_data, list):
            raise ValueError("violations_data must be a non-empty list")

        logger.info(f"Importing {len(violations_data)} violations into Neo4j")

        # Fixed: Create schema safely
        self._ensure_schema()

        records = []
        for item in violations_data:
            vid = item.get("id")
            if not vid:
                continue

            fine_min = self._parse_fine(item.get("penalty", {}).get("fine_min_vnd"))
            fine_max = self._parse_fine(item.get("penalty", {}).get("fine_max_vnd"))
            range_key = (
                f"{fine_min}_{fine_max}"
                if fine_min is not None and fine_max is not None
                else None
            )

            law = self._build_law_hierarchy(
                item.get("law_reference", {}).get("decree"),
                item.get("law_reference", {}).get("article"),
                item.get("law_reference", {}).get("clause"),
                item.get("law_reference", {}).get("point"),
            )

            add_penalties = []
            for ap in item.get("additional_penalties", []):
                ap_law = self._build_law_hierarchy(
                    ap.get("law_reference", {}).get("decree"),
                    ap.get("law_reference", {}).get("article"),
                    ap.get("law_reference", {}).get("clause"),
                    ap.get("law_reference", {}).get("point"),
                )
                key = self._generate_penalty_key(
                    {
                        "type": ap.get("type"),
                        "condition": ap.get("condition"),
                        "duration_min_months": ap.get("duration_min_months"),
                        "duration_max_months": ap.get("duration_max_months"),
                        "minus_point": ap.get("minus_point"),
                        "full_reference": ap_law["full_reference"],
                    }
                )
                add_penalties.append(
                    {
                        "penalty_key": key,
                        "type": ap.get("type", ""),
                        "condition": ap.get("condition", "") or "",
                        "description": ap.get("description", ""),
                        "duration_min": ap.get("duration_min_months"),
                        "duration_max": ap.get("duration_max_months"),
                        "minus_point": ap.get("minus_point"),
                        **ap_law,
                    }
                )

            records.append(
                {
                    "id": vid,
                    "action": item.get("canonical_action", ""),
                    "legal_desc": item.get("legal_description", ""),
                    "detailed_desc": item.get("detailed_description", ""),
                    "vehicle_type": item.get("vehicle_type", ""),
                    "fine_min": fine_min,
                    "fine_max": fine_max,
                    "range_key": range_key,
                    **law,
                    "add_penalties": add_penalties,
                }
            )

        if not records:
            raise ValueError("No valid violations to import")

        cypher = """
        UNWIND $batch AS v
        MERGE (viol:Violation {id: v.id})
        SET viol.detailed_description = v.detailed_desc,
            viol.vehicle_type = v.vehicle_type

        WITH viol, v
        MERGE (action:CanonicalAction {name: v.action})
        MERGE (viol)-[:HAS_ACTION]->(action)

        WITH viol, v
        MERGE (ld:LegalDescription {text: v.legal_desc})
        MERGE (viol)-[:HAS_LEGAL_DESC]->(ld)

        WITH viol, v
        FOREACH (_ IN CASE WHEN v.range_key IS NOT NULL THEN [1] ELSE [] END |
            MERGE (p:Penalty {range_key: v.range_key})
            SET p.fine_min_vnd = v.fine_min, p.fine_max_vnd = v.fine_max
            MERGE (viol)-[:HAS_PENALTY]->(p)
        )

        WITH viol, v
        FOREACH (_ IN CASE WHEN v.full_reference <> '' THEN [1] ELSE [] END |
            MERGE (lr:LawReference {full_reference: v.full_reference})
            SET lr.decree = v.decree, lr.article = v.article, lr.clause = v.clause, lr.point = v.point
            MERGE (viol)-[:CITED_IN]->(lr)
        )

        WITH viol, v
        FOREACH (ap IN v.add_penalties |
            MERGE (addp:AdditionalPenalty {penalty_key: ap.penalty_key})
            ON CREATE SET
                addp.type = ap.type,
                addp.condition = ap.condition,
                addp.description = ap.description,
                addp.duration_min_months = ap.duration_min,
                addp.duration_max_months = ap.duration_max,
                addp.minus_point = ap.minus_point
            MERGE (viol)-[:HAS_ADDITIONAL_PENALTY]->(addp)
            FOREACH (_ IN CASE WHEN ap.full_reference <> '' THEN [1] ELSE [] END |
                MERGE (aplr:LawReference {full_reference: ap.full_reference})
                MERGE (addp)-[:PENALTY_CITED_IN]->(aplr)
            )
        )
        """

        batch_size = 50
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self._execute_write(cypher, batch=batch)

        logger.info(f"Successfully imported {len(records)} violations")
        print(f"Success: Imported {len(records)} violations into Knowledge Graph")

    def get_violation_details(self, violation_id: str) -> dict:
        if not violation_id or not isinstance(violation_id, str):
            raise ValueError("violation_id must be a non-empty string")

        query = """
        MATCH (v:Violation {id: $id})
        OPTIONAL MATCH (v)-[:HAS_ACTION]->(a:CanonicalAction)
        OPTIONAL MATCH (v)-[:HAS_LEGAL_DESC]->(ld:LegalDescription)
        OPTIONAL MATCH (v)-[:HAS_PENALTY]->(p:Penalty)
        OPTIONAL MATCH (v)-[:CITED_IN]->(lr:LawReference)
        OPTIONAL MATCH (v)-[:HAS_ADDITIONAL_PENALTY]->(ap:AdditionalPenalty)
        OPTIONAL MATCH (ap)-[:PENALTY_CITED_IN]->(aplr:LawReference)
        OPTIONAL MATCH (v)-[:UPDATED_FROM]->(prev:Violation)
        RETURN 
            v{id: v.id, detailed_description: v.detailed_description, vehicle_type: v.vehicle_type} AS violation,
            a.name AS action,
            ld.text AS legal_description,
            p{fine_min_vnd: p.fine_min_vnd, fine_max_vnd: p.fine_max_vnd} AS penalty,
            lr{decree: lr.decree, article: lr.article, clause: lr.clause, point: lr.point} AS law_reference,
            collect(ap{
                type: ap.type,
                condition: ap.condition,
                description: ap.description,
                duration_min_months: ap.duration_min_months,
                duration_max_months: ap.duration_max_months,
                minus_point: ap.minus_point
            }) AS additional_penalties,
            prev.id AS previous_version_id
        """

        with self.driver.session() as session:
            result = session.run(query, id=violation_id).single()

        if not result or not result["violation"]:
            return {"violation": None, "previous_version_id": None}

        return {
            "violation": result["violation"],
            "action": result["action"],
            "legal_description": result["legal_description"],
            "penalty": result["penalty"],
            "law_reference": result["law_reference"],
            "additional_penalties": result["additional_penalties"] or [],
            "previous_version": (
                {"id": result["previous_version_id"]}
                if result["previous_version_id"]
                else None
            ),
        }

    def filter_violations_by_vehicle_type(
        self, violation_ids: List[str], vehicle_type: Optional[str]
    ) -> List[str]:
        if not violation_ids:
            return []
        if not vehicle_type:
            return violation_ids

        normalized = vehicle_type.lower().replace(" ", "_")
        allowed = {"ô_tô", "o_to", "xe_máy", "xe_may"}

        query = """
        UNWIND $ids AS vid
        MATCH (v:Violation {id: vid})
        WHERE toLower(v.vehicle_type) = $type OR v.vehicle_type IS NULL
        RETURN vid
        """

        with self.driver.session() as session:
            result = session.run(query, ids=violation_ids, type=normalized)
            return [record["vid"] for record in result]

    # === Bonus: Import mapping from Decree 168 → 100 ===
    def import_decree_mapping(self, mapping_json: Dict[str, Any]):
        """Import mapping from 168/2024 to 100/2019 violations"""
        if "mapping" not in mapping_json:
            raise ValueError("Invalid mapping format")

        records = []
        for v3_id, data in mapping_json["mapping"].items():
            v2_id = data.get("from_v2_id")
            if v2_id:
                records.append(
                    {"v3": v3_id, "v2": v2_id}
                )

        if not records:
            logger.info("No mappings to import")
            return

        cypher = """
        UNWIND $mappings AS m
        MERGE (v3:Violation {id: m.v3})
        MERGE (v2:Violation {id: m.v2})
        MERGE (v3)-[r:UPDATED_FROM]->(v2)
        """

        self._execute_write(cypher, mappings=records)
        logger.info(f"Imported {len(records)} decree mappings (168 → 100)")
        print(f"Success: Imported {len(records)} violation mappings")
