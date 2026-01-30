"""Retrieval evaluation metrics for e-commerce search."""

import logging
from typing import Dict, List, Set

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SparseEncoder

from qdrant_finetune.config import (
    ESCI_SCORES,
    RELEVANT_LABELS,
    WANDS_RELEVANT_LABELS,
    WANDS_SCORES,
)
from qdrant_finetune.qdrant.index import sparse_embedding_to_qdrant

logger = logging.getLogger(__name__)


def dcg_at_k(relevances: List[float], k: int) -> float:
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances: List[float], k: int = 10) -> float:
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr_at_k(relevant_ids: Set[str], retrieved_ids: List[str], k: int = 10) -> float:
    for rank, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(relevant_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(relevant_ids & set(retrieved_ids[:k])) / len(relevant_ids)


def precision_at_k(relevant_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(relevant_ids & set(retrieved_ids[:k])) / k


def _detect_label_format(ground_truth: List[Dict]) -> str:
    for q in ground_truth:
        for label in q.get("relevance", {}).values():
            if isinstance(label, int):
                return "wands"
            elif isinstance(label, str):
                return "esci"
    return "esci"


def _get_relevance_score(label, label_format: str) -> float:
    if label_format == "wands":
        return WANDS_SCORES.get(label, 0.0)
    return ESCI_SCORES.get(label, 0.0)


def _is_relevant(label, label_format: str) -> bool:
    if label_format == "wands":
        return label in WANDS_RELEVANT_LABELS
    return label in RELEVANT_LABELS


def evaluate_retrieval(
    retrieved_results: List[Dict],
    ground_truth: List[Dict],
    k_values: List[int] = [10, 50, 100],
) -> Dict[str, float]:
    """Evaluate retrieval results against ground truth."""
    label_format = _detect_label_format(ground_truth)
    gt_lookup = {q["query_id"]: q["relevance"] for q in ground_truth}

    all_ndcg = {k: [] for k in k_values}
    all_mrr = []
    all_recall = {k: [] for k in k_values}
    all_precision = {k: [] for k in k_values}
    default_label = 0 if label_format == "wands" else "I"

    for result in retrieved_results:
        query_id = result["query_id"]
        retrieved_ids = result["retrieved_ids"]
        if query_id not in gt_lookup:
            continue

        relevance_map = gt_lookup[query_id]
        relevances = [
            _get_relevance_score(relevance_map.get(pid, default_label), label_format)
            for pid in retrieved_ids
        ]
        relevant_ids = {
            pid for pid, label in relevance_map.items() if _is_relevant(label, label_format)
        }

        for k in k_values:
            all_ndcg[k].append(ndcg_at_k(relevances, k))
            all_recall[k].append(recall_at_k(relevant_ids, retrieved_ids, k))
            all_precision[k].append(precision_at_k(relevant_ids, retrieved_ids, k))
        all_mrr.append(mrr_at_k(relevant_ids, retrieved_ids, k=10))

    metrics = {"mrr@10": float(np.mean(all_mrr)) if all_mrr else 0.0}
    for k in k_values:
        metrics[f"ndcg@{k}"] = float(np.mean(all_ndcg[k])) if all_ndcg[k] else 0.0
        metrics[f"recall@{k}"] = float(np.mean(all_recall[k])) if all_recall[k] else 0.0
        metrics[f"precision@{k}"] = float(np.mean(all_precision[k])) if all_precision[k] else 0.0
    return metrics


class QdrantEvaluator:
    """Evaluate sparse encoder using Qdrant for retrieval."""

    def __init__(
        self,
        client: QdrantClient,
        model: SparseEncoder,
        collection_name: str,
        vector_name: str = "text",
    ):
        self.client = client
        self.model = model
        self.collection_name = collection_name
        self.vector_name = vector_name

    def evaluate(
        self,
        queries: List[Dict],
        k_values: List[int] = [10, 50, 100],
        show_progress: bool = True,
    ) -> Dict[str, float]:
        from tqdm import tqdm

        retrieved_results = []
        iterator = tqdm(queries, desc="Evaluating") if show_progress else queries

        for q in iterator:
            emb = self.model.encode([q["query"]])[0]
            sparse_vector = sparse_embedding_to_qdrant(emb)
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_vector,
                using=self.vector_name,
                limit=max(k_values),
                with_payload=True,
            )
            retrieved_ids = [p.payload.get("product_id", "") for p in results.points]
            retrieved_results.append({"query_id": q["query_id"], "retrieved_ids": retrieved_ids})

        return evaluate_retrieval(retrieved_results, queries, k_values)


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key in sorted(metrics.keys()):
        table.add_row(key, f"{metrics[key]:.4f}")

    console.print(table)
