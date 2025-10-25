from neo4j import GraphDatabase
from typing import List
from src.application.interfaces.i_knowledge_graph import IKnowledgeGraph # Implement interface

class Neo4jKGAdapter(IKnowledgeGraph):
    
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # Cài đặt hàm từ interface
    def add_violation_batch(self, violations_data: List[dict]):
        # Code Cypher để import JSON vào Neo4j (như ví dụ trước)
        with self.driver.session() as session:
            # session.run(...)
            print("Đã import vào Neo4j")
            pass

    # Cài đặt hàm từ interface
    def get_violation_details(self, violation_id: str) -> dict:
        # Code Cypher để lấy chi tiết vi phạm
        query = """
        MATCH (v:Violation {id: $id})
        OPTIONAL MATCH (v)-[:HAS_FINE]->(f:Fine)
        OPTIONAL MATCH (v)-[:STIPULATED_IN]->(l:LegalBasis)
        RETURN v, f, l
        """
        with self.driver.session() as session:
            result = session.run(query, id=violation_id).single()
            # (TODO: Xử lý kết quả trả về)
            # ...
            return { "violation": ..., "fine": ..., "legal_basis": ... }