"""Sparse vector indexing into Qdrant."""

import logging
from typing import Dict, List, Union

import numpy as np
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SparseEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)


def sparse_embedding_to_qdrant(
    embedding: Union[Dict, torch.Tensor, np.ndarray],
) -> models.SparseVector:
    """Convert a sparse embedding to Qdrant SparseVector format."""
    if isinstance(embedding, dict):
        indices = embedding["indices"]
        values = embedding["values"]
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        return models.SparseVector(
            indices=indices.flatten().tolist(),
            values=values.flatten().tolist(),
        )
    elif isinstance(embedding, torch.Tensor):
        if embedding.is_sparse:
            indices = embedding.coalesce().indices()[0].cpu().numpy()
            values = embedding.coalesce().values().cpu().numpy()
        else:
            embedding = embedding.cpu()
            nonzero = embedding.nonzero().squeeze(-1)
            if nonzero.dim() == 0:
                nonzero = nonzero.unsqueeze(0)
            indices = nonzero.numpy()
            values = embedding[nonzero].numpy()
        return models.SparseVector(
            indices=indices.flatten().tolist(),
            values=values.flatten().tolist(),
        )
    elif isinstance(embedding, np.ndarray):
        nonzero = np.nonzero(embedding)[0]
        values = embedding[nonzero]
        return models.SparseVector(indices=nonzero.tolist(), values=values.tolist())
    else:
        raise ValueError(f"Unsupported embedding type: {type(embedding)}")


def index_sparse_vectors(
    client: QdrantClient,
    collection_name: str,
    model: SparseEncoder,
    documents: List[Dict],
    vector_name: str = "text",
    batch_size: int = 32,
    show_progress: bool = True,
) -> int:
    """Index documents with SPLADE sparse vectors into Qdrant."""
    total_indexed = 0
    iterator = range(0, len(documents), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Indexing sparse vectors")

    for i in iterator:
        batch_docs = documents[i : i + batch_size]
        batch_texts = [doc["text"] for doc in batch_docs]

        embeddings = model.encode(
            batch_texts, convert_to_tensor=True, show_progress_bar=False
        )

        points = []
        for j, (doc, emb) in enumerate(zip(batch_docs, embeddings)):
            sparse_vector = sparse_embedding_to_qdrant(emb)
            point = models.PointStruct(
                id=i + j,
                vector={vector_name: sparse_vector},
                payload={
                    "product_id": doc["product_id"],
                    "text": doc["text"],
                    **doc.get("payload", {}),
                },
            )
            points.append(point)

        client.upsert(collection_name=collection_name, points=points)
        total_indexed += len(points)

    logger.info(f"Indexed {total_indexed} documents into {collection_name}")
    return total_indexed
