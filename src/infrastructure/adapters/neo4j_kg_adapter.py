from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable, AuthError, TransientError
from typing import List, Optional, Dict, Any
from src.application.interfaces.i_knowledge_graph import IKnowledgeGraph
import logging
import hashlib
import re
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
                session.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {uri}")
        except ServiceUnavailable as e:
            logger.error(f"Cannot connect to Neo4j at {uri}: {e}")
            raise
        except AuthError as e:
            logger.error(f"Authentication failed for Neo4j: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {e}")
            raise

    def close(self):
        if hasattr(self, "driver") and self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def _execute_with_retry(
        self, session, query: str, params: dict, max_retries: int = 3
    ):
        for attempt in range(max_retries):
            try:
                return session.execute_write(lambda tx: tx.run(query, **params))
            except TransientError as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Transient error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Query failed after {max_retries} attempts: {e}")
                    raise
            except Neo4jError as e:
                logger.error(f"Non-transient Neo4j error: {e}")
                raise

    def _create_constraints_and_indexes(self):
        queries = [
            "CREATE CONSTRAINT violation_id IF NOT EXISTS FOR (v:Violation) REQUIRE v.id IS UNIQUE",
            "CREATE CONSTRAINT action_name IF NOT EXISTS FOR (a:CanonicalAction) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT legal_text IF NOT EXISTS FOR (l:LegalDescription) REQUIRE l.text IS UNIQUE",
            "CREATE CONSTRAINT additional_penalty_key IF NOT EXISTS FOR (ap:AdditionalPenalty) REQUIRE ap.penalty_key IS UNIQUE",
            "CREATE CONSTRAINT decree_name IF NOT EXISTS FOR (d:Decree) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT article_ref IF NOT EXISTS FOR (a:Article) REQUIRE a.full_ref IS UNIQUE",
            "CREATE CONSTRAINT clause_ref IF NOT EXISTS FOR (c:Clause) REQUIRE c.full_ref IS UNIQUE",
            "CREATE CONSTRAINT law_ref IF NOT EXISTS FOR (lr:LawReference) REQUIRE lr.full_reference IS UNIQUE",
            "CREATE INDEX action_normalized IF NOT EXISTS FOR (a:CanonicalAction) ON (a.normalized_name)",
            "CREATE INDEX penalty_range IF NOT EXISTS FOR (p:Penalty) ON (p.range_key)",
            "CREATE INDEX violation_vehicle IF NOT EXISTS FOR (v:Violation) ON (v.vehicle_type)",
            "CREATE INDEX additional_penalty_type IF NOT EXISTS FOR (ap:AdditionalPenalty) ON (ap.type)",
        ]

        try:
            with self.driver.session() as session:
                for query in queries:
                    try:
                        session.run(query)
                        logger.debug(f"Executed: {query}")
                    except Neo4jError as e:
                        if "EquivalentSchemaRuleAlreadyExists" not in str(e):
                            logger.warning(f"Error creating constraint/index: {e}")
            logger.info("Successfully created/verified constraints and indexes")
            self._warmup_indexes()
        except Exception as e:
            logger.error(
                f"Error setting up constraints and indexes: {e}", exc_info=True
            )
            raise

    def _warmup_indexes(self):
        warmup_queries = [
            "MATCH (v:Violation) RETURN count(v) LIMIT 1",
            "MATCH (a:CanonicalAction) RETURN count(a) LIMIT 1",
            "MATCH (d:Decree) RETURN count(d) LIMIT 1",
        ]
        try:
            with self.driver.session() as session:
                for q in warmup_queries:
                    session.run(q)
            logger.debug("Indexes warmed up successfully")
        except Exception as e:
            logger.warning(f"Error warming up indexes: {e}")

    def _generate_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""

    def _parse_fine_amount(self, amount: Any) -> Optional[int]:
        if amount is None:
            return None
        try:
            if isinstance(amount, str):
                amount = amount.replace("_", "").replace(",", "").strip()
            return int(float(amount))
        except (ValueError, TypeError) as e:
            logger.warning(f"Cannot parse fine amount '{amount}': {e}")
            return None

    def _normalize_condition(self, condition: str) -> str:
        """Normalize empty or placeholder conditions."""
        cond = (condition or "").strip().lower()
        if cond in ["", "không", "n/a", "none", "không có"]:
            return ""
        return condition.strip()

    def _create_penalty_key(self, penalty_data: Dict[str, Any]) -> str:
        """Generate consistent full SHA256 hash for AdditionalPenalty deduplication."""
        parts = [
            str(penalty_data.get("type") or "").strip(),
            self._normalize_condition(penalty_data.get("condition")),
            str(penalty_data.get("duration_min_months") or ""),
            str(penalty_data.get("duration_max_months") or ""),
            str(penalty_data.get("full_reference") or "").strip(),
        ]
        key_string = "|".join(parts)
        return self._generate_hash(key_string)  # Full 64-char SHA256

    def _validate_hierarchy_completeness(
        self, decree: str, article: str, clause: str, point: str
    ) -> Dict[str, str]:
        decree = str(decree or "").strip()
        article = str(article or "").strip()
        clause = str(clause or "").strip()
        point = str(point or "").strip()

        decree_ref = decree
        article_ref = f"{decree}|{article}" if decree and article else ""
        clause_ref = (
            f"{decree}|{article}|{clause}" if decree and article and clause else ""
        )
        full_reference = f"{decree}|{article}|{clause}|{point}" if decree else ""

        return {
            "decree": decree,
            "article": article,
            "clause": clause,
            "point": point,
            "decree_ref": decree_ref,
            "article_ref": article_ref,
            "clause_ref": clause_ref,
            "full_reference": full_reference,
        }

    def _calculate_optimal_batch_size(self, violations: List[dict]) -> int:
        if not violations:
            return 50
        avg_penalties = sum(
            len(v.get("additional_penalties", [])) for v in violations
        ) / len(violations)
        return 25 if avg_penalties > 3 else 50 if avg_penalties > 1 else 100

    def add_violation_batch(self, violations_data: List[dict]):
        if not violations_data:
            raise ValueError("violations_data cannot be empty")
        if not isinstance(violations_data, list):
            raise ValueError("violations_data must be a list")

        logger.info(f"Starting batch import of {len(violations_data)} violations")

        try:
            self._create_constraints_and_indexes()
            transformed, skipped_count = [], 0

            for idx, item in enumerate(violations_data):
                if not isinstance(item, dict):
                    logger.warning(f"Skipping item at index {idx}: not a dictionary")
                    skipped_count += 1
                    continue

                violation_id = item.get("id")
                if not violation_id:
                    logger.warning(f"Skipping item at index {idx}: missing 'id'")
                    skipped_count += 1
                    continue

                penalty_data = item.get("penalty", {})
                fine_min = self._parse_fine_amount(penalty_data.get("fine_min_vnd"))
                fine_max = self._parse_fine_amount(penalty_data.get("fine_max_vnd"))
                range_key = (
                    f"{fine_min}_{fine_max}"
                    if fine_min is not None and fine_max is not None
                    else None
                )

                law_ref = item.get("law_reference", {})
                main_law = self._validate_hierarchy_completeness(
                    law_ref.get("decree"),
                    law_ref.get("article"),
                    law_ref.get("clause"),
                    law_ref.get("point"),
                )

                additional_penalties = []
                for add_pen in item.get("additional_penalties", []):
                    if not isinstance(add_pen, dict):
                        continue

                    try:
                        duration_min = (
                            int(add_pen.get("duration_min_months", 0))
                            if add_pen.get("duration_min_months")
                            else None
                        )
                        duration_max = (
                            int(add_pen.get("duration_max_months", 0))
                            if add_pen.get("duration_max_months")
                            else None
                        )
                    except (ValueError, TypeError):
                        duration_min = duration_max = None

                    ap_law_ref = add_pen.get("law_reference", {})
                    ap_law = self._validate_hierarchy_completeness(
                        ap_law_ref.get("decree"),
                        ap_law_ref.get("article"),
                        ap_law_ref.get("clause"),
                        ap_law_ref.get("point"),
                    )

                    penalty_info = {
                        "type": add_pen.get("type", ""),
                        "condition": add_pen.get("condition", ""),
                        "duration_min_months": duration_min,
                        "duration_max_months": duration_max,
                        "full_reference": ap_law["full_reference"],
                    }
                    penalty_key = self._create_penalty_key(penalty_info)

                    additional_penalties.append(
                        {
                            "penalty_key": penalty_key,
                            "type": add_pen.get("type", ""),
                            "condition": self._normalize_condition(
                                add_pen.get("condition")
                            ),
                            "description": add_pen.get("description", ""),
                            "duration_min_months": duration_min,
                            "duration_max_months": duration_max,
                            **ap_law,
                        }
                    )

                transformed.append(
                    {
                        "id": violation_id,
                        "canonical_action": item.get("canonical_action", ""),
                        "legal_description": item.get("legal_description", ""),
                        "detailed_description": item.get("detailed_description", ""),
                        "vehicle_type": item.get("vehicle_type", ""),
                        "fine_min": fine_min,
                        "fine_max": fine_max,
                        "range_key": range_key,
                        **main_law,
                        "additional_penalties": additional_penalties,
                    }
                )

            if not transformed:
                raise ValueError("No valid violations to import after processing")

            logger.info(
                f"Processed {len(transformed)} violations ({skipped_count} skipped)"
            )

            cypher = """
            UNWIND $violations AS v
            MERGE (viol:Violation {id: v.id})
            SET viol.detailed_description = v.detailed_description,
                viol.vehicle_type = v.vehicle_type

            WITH viol, v
            MERGE (action:CanonicalAction {name: v.canonical_action})
            SET action.normalized_name = toLower(v.canonical_action)
            MERGE (viol)-[:HAS_ACTION]->(action)

            WITH viol, v
            MERGE (legal_desc:LegalDescription {text: v.legal_description})
            MERGE (viol)-[:HAS_LEGAL_DESC]->(legal_desc)

            WITH viol, v
            FOREACH (_ IN CASE WHEN v.range_key IS NOT NULL THEN [1] ELSE [] END |
                MERGE (penalty:Penalty {range_key: v.range_key})
                SET penalty.fine_min_vnd = v.fine_min,
                    penalty.fine_max_vnd = v.fine_max
                MERGE (viol)-[:HAS_PENALTY]->(penalty)
            )

            WITH viol, v
            CALL {
                WITH v
                FOREACH (_ IN CASE WHEN v.decree_ref <> '' THEN [1] ELSE [] END |
                    MERGE (d:Decree {name: v.decree})
                )
                FOREACH (_ IN CASE WHEN v.article_ref <> '' THEN [1] ELSE [] END |
                    MERGE (d:Decree {name: v.decree})
                    MERGE (a:Article {full_ref: v.article_ref})
                    ON CREATE SET a.name = v.article, a.decree = v.decree
                    MERGE (a)-[:UNDER_DECREE]->(d)
                )
                FOREACH (_ IN CASE WHEN v.clause_ref <> '' THEN [1] ELSE [] END |
                    MERGE (a:Article {full_ref: v.article_ref})
                    MERGE (c:Clause {full_ref: v.clause_ref})
                    ON CREATE SET c.name = v.clause, c.decree = v.decree, c.article = v.article
                    MERGE (c)-[:UNDER_ARTICLE]->(a)
                )
                FOREACH (_ IN CASE WHEN v.full_reference <> '' THEN [1] ELSE [] END |
                    MERGE (lr:LawReference {full_reference: v.full_reference})
                    SET lr.decree = v.decree, lr.article = v.article,
                        lr.clause = v.clause, lr.point = v.point
                    FOREACH (_ IN CASE WHEN v.clause_ref <> '' THEN [1] ELSE [] END |
                        MERGE (c:Clause {full_ref: v.clause_ref})
                        MERGE (lr)-[:UNDER_CLAUSE]->(c)
                    )
                )
            }

            WITH viol, v
            FOREACH (_ IN CASE WHEN v.full_reference <> '' THEN [1] ELSE [] END |
                MERGE (lr:LawReference {full_reference: v.full_reference})
                MERGE (viol)-[:CITED_IN]->(lr)
            )

            WITH viol, v
            FOREACH (ap IN v.additional_penalties |
                MERGE (add_penalty:AdditionalPenalty {penalty_key: ap.penalty_key})
                ON CREATE SET 
                    add_penalty.type = ap.type,
                    add_penalty.condition = ap.condition,
                    add_penalty.description = ap.description,
                    add_penalty.duration_min_months = ap.duration_min_months,
                    add_penalty.duration_max_months = ap.duration_max_months
                MERGE (viol)-[:HAS_ADDITIONAL_PENALTY]->(add_penalty)
                
                FOREACH (_ IN CASE WHEN ap.full_reference <> '' THEN [1] ELSE [] END |
                    MERGE (ap_lr:LawReference {full_reference: ap.full_reference})
                    SET ap_lr.decree = ap.decree,
                        ap_lr.article = ap.article,
                        ap_lr.clause = ap.clause,
                        ap_lr.point = ap.point
                    MERGE (add_penalty)-[:PENALTY_CITED_IN]->(ap_lr)
                )
            )
            """

            batch_size = self._calculate_optimal_batch_size(transformed)
            logger.info(f"Using batch size: {batch_size}")
            total_imported = 0
            failed_batches = []

            with self.driver.session() as session:
                for i in range(0, len(transformed), batch_size):
                    batch = transformed[i : i + batch_size]
                    batch_num = i // batch_size + 1
                    try:
                        self._execute_with_retry(session, cypher, {"violations": batch})
                        total_imported += len(batch)
                        logger.debug(
                            f"Imported batch {batch_num}: {len(batch)} violations"
                        )
                    except Neo4jError as e:
                        logger.error(f"Error importing batch {batch_num}: {e}")
                        failed_batches.append(batch_num)

            if failed_batches:
                logger.warning(
                    f"Import completed with errors. Failed batches: {failed_batches}"
                )
                print(
                    f"Warning: Imported {total_imported}/{len(transformed)} violations. Failed batches: {failed_batches}"
                )
            else:
                logger.info(f"Successfully imported {total_imported} violations")
                print(f"Success: Imported {total_imported} violations into Neo4j KG")

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Neo4jError as e:
            logger.error(f"Neo4j error during batch import: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during batch import: {e}", exc_info=True)
            raise

    def cleanup_additional_penalties(self):
        """Run once to merge duplicate AdditionalPenalty nodes."""
        logger.info("Starting AdditionalPenalty deduplication cleanup...")
        try:
            with self.driver.session() as session:
                # Add penalty_key to nodes missing it
                result = session.run(
                    """
                    MATCH (ap:AdditionalPenalty)
                    WHERE ap.penalty_key IS NULL
                    RETURN ap
                """
                )
                count = 0
                for record in result:
                    ap = record["ap"]
                    data = {
                        "type": ap.get("type"),
                        "condition": ap.get("condition"),
                        "duration_min_months": ap.get("duration_min_months"),
                        "duration_max_months": ap.get("duration_max_months"),
                        "full_reference": ap.get("full_reference", ""),
                    }
                    key = self._create_penalty_key(data)
                    session.run(
                        """
                        MATCH (ap:AdditionalPenalty)
                        WHERE elementId(ap) = $id
                        SET ap.penalty_key = $key
                    """,
                        id=ap.element_id,
                        key=key,
                    )
                    count += 1
                logger.info(f"Added penalty_key to {count} nodes.")

                # Merge duplicates
                merged = session.run(
                    """
                    MATCH (ap:AdditionalPenalty)
                    WITH ap.type AS t, COALESCE(ap.condition, '') AS c,
                         ap.duration_min_months AS min_m, ap.duration_max_months AS max_m,
                         ap.description AS d, collect(ap) AS nodes
                    WHERE size(nodes) > 1
                    CALL {
                      WITH nodes
                      UNWIND nodes AS n
                      WITH n ORDER BY n.penalty_key DESC NULLS LAST LIMIT 1
                      RETURN n AS canonical
                    }
                    WITH canonical, nodes
                    UNWIND [n IN nodes WHERE n <> canonical] AS dup
                    MATCH (v:Violation)-[r:HAS_ADDITIONAL_PENALTY]->(dup)
                    MERGE (v)-[:HAS_ADDITIONAL_PENALTY]->(canonical)
                    DELETE r
                    DETACH DELETE dup
                   infection
                    RETURN count(*) AS merged_count
                """
                ).single()
                merged_count = merged["merged_count"] if merged else 0
                logger.info(f"Merged {merged_count} duplicate AdditionalPenalty nodes.")
                print(f"Success: Cleanup complete. Merged {merged_count} duplicates.")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}", exc_info=True)
            raise

    # ... [rest of your methods: filter_violations_by_vehicle_type, search_violations_by_action, etc.] ...
    # (All other methods remain unchanged)

    def filter_violations_by_vehicle_type(
        self, violation_ids: List[str], vehicle_type: Optional[str]
    ) -> List[str]:
        if not violation_ids:
            logger.debug("Empty violation_ids list provided")
            return []
        if not vehicle_type:
            logger.debug("No vehicle type filter, returning all")
            return violation_ids

        logger.debug(
            f"Filtering {len(violation_ids)} violations by vehicle_type: {vehicle_type}"
        )

        query = """
        MATCH (v:Violation)
        WHERE v.id IN $violation_ids
          AND toLower(v.vehicle_type) CONTAINS toLower($vehicle_type)
        RETURN v.id AS violation_id
        ORDER BY v.id
        """

        try:
            with self.driver.session() as session:
                result = session.execute_read(
                    lambda tx: tx.run(
                        query, violation_ids=violation_ids, vehicle_type=vehicle_type
                    )
                )
                filtered_ids = [record["violation_id"] for record in result]
            logger.info(
                f"Filtered {len(filtered_ids)}/{len(violation_ids)} by '{vehicle_type}'"
            )
            return filtered_ids
        except Neo4jError as e:
            logger.error(f"Neo4j error filtering by vehicle type: {e}", exc_info=True)
            logger.warning("Returning all violations due to filter error")
            return violation_ids
        except Exception as e:
            logger.error(f"Unexpected error filtering: {e}", exc_info=True)
            return violation_ids

    def search_violations_by_action(
        self, action_keyword: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        if not action_keyword:
            raise ValueError("action_keyword cannot be empty")
        if not (1 <= limit <= 1000):
            raise ValueError("limit must be between 1 and 1000")

        logger.debug(f"Searching violations by action: {action_keyword}")

        query = """
        MATCH (a:CanonicalAction)
        WHERE toLower(a.name) CONTAINS toLower($keyword)
        MATCH (v:Violation)-[:HAS_ACTION]->(a)
        OPTIONAL MATCH (v)-[:HAS_PENALTY]->(p:Penalty)
        RETURN v.id AS id, a.name AS action, v.detailed_description AS description,
               v.vehicle_type AS vehicle_type, p.fine_min_vnd AS fine_min, p.fine_max_vnd AS fine_max
        ORDER BY p.fine_max_vnd DESC
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.execute_read(
                    lambda tx: tx.run(query, keyword=action_keyword, limit=limit)
                )
                violations = [
                    {
                        "id": r["id"],
                        "action": r["action"],
                        "description": r["description"],
                        "vehicle_type": r["vehicle_type"],
                        "fine_min_vnd": r["fine_min"],
                        "fine_max_vnd": r["fine_max"],
                    }
                    for r in result
                ]
            logger.info(
                f"Found {len(violations)} violations matching '{action_keyword}'"
            )
            return violations
        except Neo4jError as e:
            logger.error(f"Neo4j error searching by action: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise

    def get_violations_by_decree(
        self, decree_name: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        if not decree_name:
            raise ValueError("decree_name cannot be empty")
        if not re.match(r"^[\w\-/]+$", decree_name):
            raise ValueError(f"Invalid decree name format: {decree_name}")
        if not (1 <= limit <= 1000):
            raise ValueError("limit must be between 1 and 1000")

        logger.debug(f"Getting violations under decree: {decree_name}")

        query = """
        MATCH (decree:Decree {name: $decree_name})
        MATCH (decree)<-[:UNDER_DECREE]-(article:Article)
        MATCH (article)<-[:UNDER_ARTICLE]-(clause:Clause)
        MATCH (clause)<-[:UNDER_CLAUSE]-(lr:LawReference)
        MATCH (v:Violation)-[:CITED_IN]->(lr)
        OPTIONAL MATCH (v)-[:HAS_ACTION]->(a:CanonicalAction)
        OPTIONAL MATCH (v)-[:HAS_PENALTY]->(p:Penalty)
        RETURN v.id AS id, a.name AS action, v.detailed_description AS description,
               article.name AS article, clause.name AS clause, lr.point AS point,
               p.fine_min_vnd AS fine_min, p.fine_max_vnd AS fine_max
        ORDER BY article.name, clause.name, lr.point
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.execute_read(
                    lambda tx: tx.run(query, decree_name=decree_name, limit=limit)
                )
                violations = [
                    {
                        "id": r["id"],
                        "action": r["action"],
                        "description": r["description"],
                        "article": r["article"],
                        "clause": r["clause"],
                        "point": r["point"],
                        "fine_min_vnd": r["fine_min"],
                        "fine_max_vnd": r["fine_max"],
                    }
                    for r in result
                ]
            logger.info(
                f"Found {len(violations)} violations under decree '{decree_name}'"
            )
            return violations
        except Neo4jError as e:
            logger.error(
                f"Neo4j error getting violations by decree: {e}", exc_info=True
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise

    def get_additional_penalty_usage(
        self, penalty_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        logger.debug(f"Getting additional penalty usage (type: {penalty_type})")

        if penalty_type:
            query = """
            MATCH (ap:AdditionalPenalty {type: $penalty_type})
            MATCH (v:Violation)-[:HAS_ADDITIONAL_PENALTY]->(ap)
            WITH ap, count(v) AS violation_count
            RETURN ap.type AS type, ap.condition AS condition, ap.description AS description,
                   ap.duration_min_months AS duration_min, ap.duration_max_months AS duration_max,
                   violation_count
            ORDER BY violation_count DESC
            """
            params = {"penalty_type": penalty_type}
        else:
            query = """
            MATCH (ap:AdditionalPenalty)
            MATCH (v:Violation)-[:HAS_ADDITIONAL_PENALTY]->(ap)
            WITH ap, count(v) AS violation_count
            RETURN ap.type AS type, ap.condition AS condition, ap.description AS description,
                   ap.duration_min_months AS duration_min, ap.duration_max_months AS duration_max,
                   violation_count
            ORDER BY violation_count DESC
            LIMIT 50
            """
            params = {}

        try:
            with self.driver.session() as session:
                result = session.execute_read(lambda tx: tx.run(query, **params))
                penalties = [
                    {
                        "type": r["type"],
                        "condition": r["condition"],
                        "description": r["description"],
                        "duration_min_months": r["duration_min"],
                        "duration_max_months": r["duration_max"],
                        "violation_count": r["violation_count"],
                    }
                    for r in result
                ]
            logger.info(f"Found {len(penalties)} additional penalty clusters")
            return penalties
        except Neo4jError as e:
            logger.error(f"Neo4j error getting penalty usage: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise

    def get_statistics(self) -> Dict[str, Any]:
        logger.debug("Retrieving knowledge graph statistics")

        query = """
        MATCH (v:Violation) WITH count(v) AS violation_count
        MATCH (a:CanonicalAction) WITH violation_count, count(a) AS action_count
        MATCH (p:Penalty) WITH violation_count, action_count, count(p) AS penalty_count
        MATCH (ap:AdditionalPenalty) WITH violation_count, action_count, penalty_count, count(ap) AS additional_penalty_count
        MATCH (decree:Decree) WITH violation_count, action_count, penalty_count, additional_penalty_count, count(decree) AS decree_count
        MATCH (article:Article) WITH violation_count, action_count, penalty_count, additional_penalty_count, decree_count, count(article) AS article_count
        MATCH (clause:Clause) WITH violation_count, action_count, penalty_count, additional_penalty_count, decree_count, article_count, count(clause) AS clause_count
        MATCH (lr:LawReference) WITH violation_count, action_count, penalty_count, additional_penalty_count, decree_count, article_count, clause_count, count(lr) AS law_ref_count
        RETURN violation_count, action_count, penalty_count, additional_penalty_count, 
               decree_count, article_count, clause_count, law_ref_count
        """

        try:
            with self.driver.session() as session:
                result = session.execute_read(lambda tx: tx.run(query).single())
            if result:
                stats = {
                    "violations": result["violation_count"],
                    "canonical_actions": result["action_count"],
                    "penalty_ranges": result["penalty_count"],
                    "additional_penalties_clustered": result[
                        "additional_penalty_count"
                    ],
                    "legal_hierarchy": {
                        "decrees": result["decree_count"],
                        "articles": result["article_count"],
                        "clauses": result["clause_count"],
                        "law_references": result["law_ref_count"],
                    },
                }
                logger.info(f"Retrieved KG statistics: {stats}")
                return stats
            else:
                logger.warning("No statistics found")
                return {
                    k: 0
                    for k in [
                        "violations",
                        "canonical_actions",
                        "penalty_ranges",
                        "additional_penalties_clustered",
                    ]
                }
        except Neo4jError as e:
            logger.error(f"Neo4j error retrieving statistics: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise

    def verify_clustering(self) -> Dict[str, Any]:
        logger.debug("Verifying additional penalty clustering")

        query = """
        MATCH (v:Violation)-[:HAS_ADDITIONAL_PENALTY]->(ap:AdditionalPenalty)
        WITH count(DISTINCT v) AS total_violations_with_penalties,
             count(DISTINCT ap) AS unique_penalties,
             count(*) AS total_relationships
        MATCH (ap:AdditionalPenalty)<-[:HAS_ADDITIONAL_PENALTY]-(v:Violation)
        WITH total_violations_with_penalties, unique_penalties, total_relationships,
             ap.penalty_key AS penalty_key, ap.type AS type, count(v) AS violation_count
        ORDER BY violation_count DESC LIMIT 5
        WITH total_violations_with_penalties, unique_penalties, total_relationships,
             collect({penalty_key: penalty_key, type: type, violation_count: violation_count}) AS top_clusters
        RETURN total_violations_with_penalties, unique_penalties, total_relationships, top_clusters,
               CASE WHEN unique_penalties > 0 THEN toFloat(total_relationships) / unique_penalties ELSE 0.0 END AS avg_violations_per_cluster
        """

        try:
            with self.driver.session() as session:
                result = session.execute_read(lambda tx: tx.run(query).single())
            if result and result["unique_penalties"] > 0:
                info = {
                    "total_violations_with_penalties": result[
                        "total_violations_with_penalties"
                    ],
                    "unique_penalty_clusters": result["unique_penalties"],
                    "total_penalty_relationships": result["total_relationships"],
                    "avg_violations_per_cluster": round(
                        result["avg_violations_per_cluster"], 2
                    ),
                    "top_clusters": result["top_clusters"],
                    "clustering_effectiveness": (
                        "Good" if result["avg_violations_per_cluster"] > 1.5 else "Poor"
                    ),
                }
                logger.info(
                    f"Clustering: {info['clustering_effectiveness']} (avg {info['avg_violations_per_cluster']})"
                )
                return info
            else:
                logger.warning("No clustering data")
                return {
                    k: 0
                    for k in [
                        "total_violations_with_penalties",
                        "unique_penalty_clusters",
                        "total_penalty_relationships",
                        "avg_violations_per_cluster",
                    ]
                } | {"top_clusters": [], "clustering_effectiveness": "No data"}
        except Neo4jError as e:
            logger.error(f"Neo4j error verifying clustering: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise

    def clear_all_data(self):
        logger.warning("Clearing all data from knowledge graph")
        query = "MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 1000 ROWS"
        try:
            with self.driver.session() as session:
                session.run(query)
            logger.info("All data cleared")
            print("Success: All data cleared from Neo4j KG")
        except Neo4jError as e:
            logger.error(f"Neo4j error clearing data: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise

    def get_health_check(self) -> Dict[str, Any]:
        logger.debug("Performing health check")
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
                result = session.execute_read(
                    lambda tx: tx.run(
                        "MATCH (n) RETURN count(n) AS total_nodes"
                    ).single()
                )
                total_nodes = result["total_nodes"] if result else 0
                status = "healthy" if total_nodes > 0 else "empty"
                health = {
                    "status": status,
                    "connection": "ok",
                    "total_nodes": total_nodes,
                    "has_data": total_nodes > 0,
                }
                logger.info(f"Health check: {status}")
                return health
        except ServiceUnavailable as e:
            logger.error(f"Service unavailable: {e}")
            return {
                "status": "unhealthy",
                "connection": "failed",
                "error": str(e),
                "total_nodes": 0,
                "has_data": False,
            }
        except Neo4jError as e:
            logger.error(f"Neo4j error: {e}")
            return {
                "status": "unhealthy",
                "connection": "error",
                "error": str(e),
                "total_nodes": 0,
                "has_data": False,
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {
                "status": "unhealthy",
                "connection": "error",
                "error": str(e),
                "total_nodes": 0,
                "has_data": False,
            }

    def get_violation_details(self, violation_id: str) -> Dict[str, Any]:
        if not violation_id or not isinstance(violation_id, str):
            raise ValueError("violation_id must be a non-empty string")

        logger.debug(f"Retrieving details for violation: {violation_id}")

        query = """
        MATCH (v:Violation {id: $id})
        OPTIONAL MATCH (v)-[:HAS_ACTION]->(a:CanonicalAction)
        OPTIONAL MATCH (v)-[:HAS_LEGAL_DESC]->(ld:LegalDescription)
        OPTIONAL MATCH (v)-[:HAS_PENALTY]->(p:Penalty)
        OPTIONAL MATCH (v)-[:CITED_IN]->(lr:LawReference)
        OPTIONAL MATCH (lr)-[:UNDER_CLAUSE]->(clause:Clause)
        OPTIONAL MATCH (clause)-[:UNDER_ARTICLE]->(article:Article)
        OPTIONAL MATCH (article)-[:UNDER_DECREE]->(decree:Decree)
        OPTIONAL MATCH (v)-[:HAS_ADDITIONAL_PENALTY]->(ap:AdditionalPenalty)
        OPTIONAL MATCH (ap)-[:PENALTY_CITED_IN]->(ap_lr:LawReference)
        OPTIONAL MATCH (ap_lr)-[:UNDER_CLAUSE]->(ap_clause:Clause)
        OPTIONAL MATCH (ap_clause)-[:UNDER_ARTICLE]->(ap_article:Article)
        OPTIONAL MATCH (ap_article)-[:UNDER_DECREE]->(ap_decree:Decree)
        RETURN v, a, ld, p, lr, clause, article, decree,
               collect(DISTINCT {
                   penalty: ap, law_ref: ap_lr, clause: ap_clause,
                   article: ap_article, decree: ap_decree
               }) AS additional_penalties
        """

        try:
            with self.driver.session() as session:
                result = session.execute_read(
                    lambda tx: tx.run(query, id=violation_id).single()
                )

            if not result:
                logger.warning(f"Violation '{violation_id}' not found")
                return {
                    k: None
                    for k in [
                        "violation",
                        "action",
                        "legal_description",
                        "penalty",
                        "law_reference",
                    ]
                } | {"additional_penalties": []}

            v_node = result["v"]
            violation = (
                {
                    "id": v_node["id"],
                    "detailed_description": v_node.get("detailed_description"),
                    "vehicle_type": v_node.get("vehicle_type"),
                }
                if v_node
                else None
            )

            action = result["a"]["name"] if result["a"] else None
            legal_description = result["ld"]["text"] if result["ld"] else None

            penalty = (
                {
                    "fine_min_vnd": result["p"]["fine_min_vnd"],
                    "fine_max_vnd": result["p"]["fine_max_vnd"],
                }
                if result["p"]
                else None
            )

            law_ref = result["lr"]
            clause = result["clause"]
            article = result["article"]
            decree = result["decree"]

            law_reference = (
                {
                    "decree": decree["name"] if decree else law_ref.get("decree"),
                    "article": article["name"] if article else law_ref.get("article"),
                    "clause": clause["name"] if clause else law_ref.get("clause"),
                    "point": law_ref["point"] if law_ref else None,
                    "hierarchy": {
                        "decree_name": decree["name"] if decree else None,
                        "article_ref": article["full_ref"] if article else None,
                        "clause_ref": clause["full_ref"] if clause else None,
                    },
                }
                if law_ref
                else None
            )

            additional_penalties = []
            for ap_data in result["additional_penalties"]:
                ap = ap_data["penalty"]
                if not ap or not ap.get("penalty_key"):
                    continue
                ap_dict = {
                    "type": ap["type"],
                    "condition": ap["condition"],
                    "description": ap["description"],
                    "duration_min_months": ap["duration_min_months"],
                    "duration_max_months": ap["duration_max_months"],
                }
                ap_lr = ap_data["law_ref"]
                if ap_lr:
                    ap_dict["law_reference"] = {
                        "decree": (
                            ap_data["decree"]["name"]
                            if ap_data["decree"]
                            else ap_lr.get("decree")
                        ),
                        "article": (
                            ap_data["article"]["name"]
                            if ap_data["article"]
                            else ap_lr.get("article")
                        ),
                        "clause": (
                            ap_data["clause"]["name"]
                            if ap_data["clause"]
                            else ap_lr.get("clause")
                        ),
                        "point": ap_lr["point"],
                    }
                additional_penalties.append(ap_dict)

            result_dict = {
                "violation": violation,
                "action": action,
                "legal_description": legal_description,
                "penalty": penalty,
                "law_reference": law_reference,
                "additional_penalties": additional_penalties,
            }
            logger.info(f"Retrieved details for violation '{violation_id}'")
            return result_dict
        except Neo4jError as e:
            logger.error(f"Neo4j error retrieving violation: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise
