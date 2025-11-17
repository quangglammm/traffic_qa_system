#!/usr/bin/env python3
"""
Script to import traffic violation data into both Knowledge Graph and Vector Store.
This script should be run before using the QA system.
"""

import sys
import json
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
        uri=settings.NEO4J_URI, user=settings.NEO4J_USER, password=settings.NEO4J_PASS
    )

    vs_adapter = ChromaVSAdapter(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
    )

    embedding_adapter = EmbeddingAdapter(
        backend_type=settings.EMBEDDING_BACKEND_TYPE,
        model_name=settings.EMBEDDING_MODEL_NAME,
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL,
    )

    # Load data files
    data_dir = Path(__file__).parent.parent / "data"
    v2_file = data_dir / "violations_v2.json"
    v3_file = data_dir / "violations_v3.json"
    mapping_file = data_dir / "mapping_output.json"

    for f in [v2_file, v3_file, mapping_file]:
        if not f.exists():
            logger.error(f"Data file not found: {f}")
            return

    logger.info("Loading data files...")
    with open(v2_file, "r", encoding="utf-8") as f:
        v2_data = json.load(f)
    with open(v3_file, "r", encoding="utf-8") as f:
        v3_data = json.load(f)
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    logger.info(
        f"Loaded: v2={len(v2_data)}, v3={len(v3_data)}, mappings={len(mapping_data.get('mapping', {}))}"
    )

    # Step 1: Import into Knowledge Graph
    logger.info("Step 1: Importing into Knowledge Graph...")
    try:
        # Import v2 (old decree)
        kg_adapter.add_violation_batch(v2_data)
        logger.info("✓ Imported v2 violations")

        # Import v3 (current decree)
        kg_adapter.add_violation_batch(v3_data)
        logger.info("✓ Imported v3 violations")

        # Import mappings (links v3 to v2)
        kg_adapter.import_decree_mapping(mapping_data)
        logger.info("✓ Imported mappings")
    except Exception as e:
        logger.error(f"✗ Error importing into Knowledge Graph: {e}")
        return

    # Step 2: Prepare data for Vector Store (only v3, as current)
    logger.info("Step 2: Preparing embeddings for Vector Store (v3 only)...")
    violation_ids = []
    descriptions = []
    metadatas = []

    for item in v3_data:
        violation_id = item.get("id")
        description = f"{item.get('canonical_action', '')} {item.get('detailed_description', '')}".strip()

        if violation_id and description:
            violation_ids.append(violation_id)
            descriptions.append(description)
            metadatas.append(
                {"vehicle_type": item.get("vehicle_type", ""), "version": "168/2024"}
            )

    logger.info(f"Prepared {len(violation_ids)} v3 violations for embedding")

    # Step 3: Generate embeddings in batches
    logger.info("Step 3: Generating embeddings...")
    batch_size = 32  # Adjustable; increase if model/memory allows
    all_embeddings = []

    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(descriptions) + batch_size - 1)//batch_size}..."
        )
        batch_embeddings = embedding_adapter.embed_batch(batch)
        all_embeddings.extend(batch_embeddings)

    logger.info("✓ Successfully generated embeddings")

    # Step 4: Import into Vector Store
    logger.info("Step 4: Importing into Vector Store...")
    try:
        # Assume vs_adapter.add_documents(ids, embeddings, texts, metadatas)
        # Update adapter if needed to support metadatas
        vs_adapter.add_documents(violation_ids, all_embeddings, descriptions, metadatas)
        logger.info("✓ Successfully imported into Vector Store")
    except Exception as e:
        logger.error(f"✗ Error importing into Vector Store: {e}")
        return

    # Cleanup
    kg_adapter.close()

    logger.info("=" * 50)
    logger.info("✓ Data import completed successfully!")
    logger.info(
        f"  - Imported v2: {len(v2_data)}, v3: {len(v3_data)} into Knowledge Graph"
    )
    logger.info(f"  - Imported mappings: {len(mapping_data.get('mapping', {}))}")
    logger.info(f"  - Imported {len(violation_ids)} v3 violations into Vector Store")
    logger.info("=" * 50)


if __name__ == "__main__":
    try:
        load_data()
    except KeyboardInterrupt:
        logger.info("\nImport interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
