"""Qdrant client wrapper for sparse vector operations."""

import logging
import os
from typing import Optional

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


def get_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> QdrantClient:
    """Get a Qdrant client instance."""
    url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = api_key or os.environ.get("QDRANT_API_KEY")

    client = QdrantClient(url=url, api_key=api_key if api_key else None, timeout=60)
    logger.info(f"Connected to Qdrant at {url}")
    return client


def create_sparse_collection(
    client: QdrantClient,
    collection_name: str,
    vector_name: str = "text",
    on_disk: bool = False,
    recreate: bool = False,
) -> None:
    """Create a Qdrant collection for SPLADE sparse vectors."""
    if recreate:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={},
            sparse_vectors_config={
                vector_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=on_disk),
                )
            },
        )
        logger.info(f"Created sparse collection: {collection_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info(f"Collection {collection_name} already exists")
        else:
            raise


def delete_collection(client: QdrantClient, collection_name: str) -> None:
    """Delete a Qdrant collection."""
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
    except Exception as e:
        logger.warning(f"Failed to delete collection {collection_name}: {e}")


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a collection exists."""
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False
