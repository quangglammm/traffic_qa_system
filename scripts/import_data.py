#!/usr/bin/env python3
"""
Script to import traffic violation data into both Knowledge Graph and Vector Store.
This script should be run before using the QA system.
"""

import sys
import json
import os
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.config import settings
from src.infrastructure.adapters.neo4j_kg_adapter import Neo4jKGAdapter
from src.infrastructure.adapters.chroma_vs_adapter import ChromaVSAdapter
from src.infrastructure.adapters.embedding_adapter import EmbeddingAdapter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load traffic violation data from JSON files into KG and Vector Store"""

    # Initialize adapters
    logger.info("Initializing adapters...")
    kg_adapter = Neo4jKGAdapter(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASS
    )

    vs_adapter = ChromaVSAdapter(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        persist_directory=settings.CHROMA_PERSIST_DIRECTORY
    )

    embedding_adapter = EmbeddingAdapter(
        backend_type=settings.EMBEDDING_BACKEND_TYPE,
        model_name=settings.EMBEDDING_MODEL_NAME,
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL
    )

    # Load violations data
    data_dir = Path(__file__).parent.parent / "data"
    violations_file = data_dir / "violations.json"

    if not violations_file.exists():
        logger.error(f"Violations data file not found: {violations_file}")
        return

    logger.info(f"Loading data from {violations_file}...")
    with open(violations_file, 'r', encoding='utf-8') as f:
        violations_data = json.load(f)

    logger.info(f"Found {len(violations_data)} violations to import")

    # Step 1: Import into Knowledge Graph
    logger.info("Step 1: Importing violations into Knowledge Graph...")
    try:
        kg_adapter.add_violation_batch(violations_data)
        logger.info("✓ Successfully imported into Knowledge Graph")
    except Exception as e:
        logger.error(f"✗ Error importing into Knowledge Graph: {e}")
        return

    # Step 2: Prepare data for Vector Store
    logger.info("Step 2: Preparing embeddings for Vector Store...")
    violation_ids = []
    descriptions = []

    for item in violations_data:
        violation_id = item.get("id")
        # Use both action and description for better semantic search
        description = f"{item.get('hanh_vi_chung', '')} {item.get('mo_ta_luat', '')}".strip()

        if violation_id and description:
            violation_ids.append(violation_id)
            descriptions.append(description)

    logger.info(f"Prepared {len(violation_ids)} violations for embedding")

    # Step 3: Generate embeddings in batches
    logger.info("Step 3: Generating embeddings...")
    batch_size = 32  # Process in batches to avoid memory issues
    all_embeddings = []

    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(descriptions) + batch_size - 1)//batch_size}...")
        batch_embeddings = embedding_adapter.embed_batch(batch)
        all_embeddings.extend(batch_embeddings)

    logger.info("✓ Successfully generated embeddings")

    # Step 4: Import into Vector Store
    logger.info("Step 4: Importing into Vector Store...")
    try:
        vs_adapter.add_violations(violation_ids, all_embeddings, descriptions)
        logger.info("✓ Successfully imported into Vector Store")
    except Exception as e:
        logger.error(f"✗ Error importing into Vector Store: {e}")
        return

    # Cleanup
    kg_adapter.close()

    logger.info("=" * 50)
    logger.info("✓ Data import completed successfully!")
    logger.info(f"  - Imported {len(violations_data)} violations into Knowledge Graph")
    logger.info(f"  - Imported {len(violation_ids)} violations into Vector Store")
    logger.info("=" * 50)

if __name__ == "__main__":
    try:
        load_data()
    except KeyboardInterrupt:
        logger.info("\nImport interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

