"""Relevance feedback integration for fine-tuned SPLADE models.

Uses Qdrant's native RelevanceFeedbackQuery to re-traverse the full index
with a feedback-adjusted scoring formula, surfacing results the initial
sparse retrieval missed.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SparseEncoder

from qdrant_finetune.qdrant.index import sparse_embedding_to_qdrant

logger = logging.getLogger(__name__)


def _encode_query(model, query: str):
    """Encode a query and return a Qdrant-compatible vector.

    Returns a SparseVector for sparse encoders, or a plain list for dense.
    """
    emb = model.encode([query])[0]

    # check if the embedding looks sparse (has explicit indices/values or is very high-dim with many zeros)
    if isinstance(emb, dict) and "indices" in emb:
        return sparse_embedding_to_qdrant(emb)

    emb_array = np.asarray(emb)
    nonzero_ratio = np.count_nonzero(emb_array) / max(emb_array.size, 1)

    # if less than 50% non-zero and dim > 100, treat as sparse
    if nonzero_ratio < 0.5 and emb_array.size > 100:
        return sparse_embedding_to_qdrant(emb)

    # dense vector: return as plain list
    return emb_array.flatten().tolist()


@dataclass
class RFParams:
    """Learned parameters for the naive relevance feedback formula.

    F = a * score + sum(confidence^b * c * delta)
    """

    a: float = 0.25
    b: float = 1.0
    c: float = 0.5

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "c": self.c}

    @classmethod
    def from_dict(cls, d: dict) -> "RFParams":
        return cls(a=d["a"], b=d["b"], c=d["c"])


@dataclass
class RFConfig:
    """Configuration for relevance feedback."""

    feedback_model: str = "mixedbread-ai/mxbai-embed-large-v1"
    context_limit: int = 10
    rf_limit: int = 10
    sparse_vector_name: str = "text"
    dense_vector_name: str = "feedback"
    dense_vector_size: int = 1024
    params: RFParams = field(default_factory=RFParams)


class RelevanceFeedbackSearch:
    """Two-pass search: SPLADE retrieval then relevance feedback re-traversal."""

    def __init__(
        self,
        client: QdrantClient,
        splade_model: SparseEncoder,
        collection_name: str,
        config: Optional[RFConfig] = None,
    ):
        self.client = client
        self.splade_model = splade_model
        self.collection_name = collection_name
        self.config = config or RFConfig()
        self._feedback_model = None

    @property
    def feedback_model(self):
        if self._feedback_model is None:
            from fastembed import TextEmbedding

            self._feedback_model = TextEmbedding(self.config.feedback_model)
        return self._feedback_model

    @feedback_model.setter
    def feedback_model(self, model):
        self._feedback_model = model

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        context_limit: Optional[int] = None,
        params: Optional[RFParams] = None,
    ) -> list:
        """Two-pass search with relevance feedback.

        1. Initial SPLADE retrieval (context_limit results)
        2. Score initial results with feedback model
        3. Re-traverse index with RelevanceFeedbackQuery
        """
        limit = limit or self.config.rf_limit
        context_limit = context_limit or self.config.context_limit
        params = params or self.config.params

        # pass 1: initial sparse retrieval
        initial_results = self._initial_retrieval(query, context_limit)
        if not initial_results:
            return []

        # score with feedback model
        feedback_scores = self._score_with_feedback(query, initial_results)

        # pass 2: relevance feedback query
        rf_results = self._rf_retrieval(query, initial_results, feedback_scores, limit, params)

        return rf_results

    def search_compare(
        self,
        query: str,
        limit: Optional[int] = None,
        context_limit: Optional[int] = None,
        params: Optional[RFParams] = None,
    ) -> dict:
        """Run both vanilla and RF search, return both for comparison."""
        limit = limit or self.config.rf_limit
        context_limit = context_limit or self.config.context_limit
        params = params or self.config.params

        initial_results = self._initial_retrieval(query, limit)
        if not initial_results:
            return {"vanilla": [], "relevance_feedback": [], "query": query}

        context_results = initial_results[:context_limit]
        feedback_scores = self._score_with_feedback(query, context_results)
        rf_results = self._rf_retrieval(query, context_results, feedback_scores, limit, params)

        return {
            "query": query,
            "vanilla": [
                {"id": p.id, "score": p.score, "payload": p.payload}
                for p in initial_results
            ],
            "relevance_feedback": [
                {"id": p.id, "score": p.score, "payload": p.payload}
                for p in rf_results
            ],
        }

    def _initial_retrieval(self, query: str, limit: int) -> list:
        query_vector = _encode_query(self.splade_model, query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=self.config.sparse_vector_name,
            limit=limit,
            with_payload=True,
        )
        return results.points

    def _score_with_feedback(self, query: str, results: list) -> list[float]:
        query_emb = list(self.feedback_model.embed([query]))[0]

        scores = []
        for point in results:
            doc_text = point.payload.get("text", "")
            doc_emb = list(self.feedback_model.embed([doc_text]))[0]
            score = float(np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8
            ))
            scores.append(score)

        return scores

    def _rf_retrieval(
        self,
        query: str,
        initial_results: list,
        feedback_scores: list[float],
        limit: int,
        params: RFParams,
    ) -> list:
        query_vector = _encode_query(self.splade_model, query)

        feedback_items = [
            models.FeedbackItem(example=point.id, score=score)
            for point, score in zip(initial_results, feedback_scores)
        ]

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.RelevanceFeedbackQuery(
                relevance_feedback=models.RelevanceFeedbackInput(
                    target=query_vector,
                    feedback=feedback_items,
                    strategy=models.NaiveFeedbackStrategy(
                        naive=models.NaiveFeedbackStrategyParams(
                            a=params.a,
                            b=params.b,
                            c=params.c,
                        )
                    ),
                )
            ),
            using=self.config.sparse_vector_name,
            with_payload=True,
            limit=limit,
        )
        return results.points


def train_rf_params(
    client: QdrantClient,
    splade_model: SparseEncoder,
    collection_name: str,
    queries: List[str],
    config: Optional[RFConfig] = None,
) -> RFParams:
    """Train relevance feedback formula parameters (a, b, c) using qdrant-relevance-feedback.

    Requires: pip install qdrant-relevance-feedback
    """
    try:
        from qdrant_relevance_feedback import RelevanceFeedback
        from qdrant_relevance_feedback.feedback import FastembedFeedback
        from qdrant_relevance_feedback.retriever import QdrantRetriever
    except ImportError:
        raise ImportError(
            "qdrant-relevance-feedback is required for parameter training. "
            "Install it with: pip install qdrant-relevance-feedback"
        )

    config = config or RFConfig()

    rf = RelevanceFeedback(
        retriever=QdrantRetriever(config.sparse_vector_name),
        feedback=FastembedFeedback(config.feedback_model),
        client=client,
        collection_name=collection_name,
        vector_name=config.sparse_vector_name,
        payload_key="text",
    )

    learned = rf.train(queries=queries, limit=25)
    logger.info(f"Trained RF params: {learned}")

    return RFParams(
        a=learned.get("a", 0.25),
        b=learned.get("b", 1.0),
        c=learned.get("c", 0.5),
    )


def evaluate_rf(
    client: QdrantClient,
    splade_model: SparseEncoder,
    collection_name: str,
    queries: List[Dict],
    config: Optional[RFConfig] = None,
    k_values: List[int] = [10, 50, 100],
) -> dict:
    """Evaluate relevance feedback vs vanilla retrieval.

    Returns metrics for both vanilla and RF retrieval, plus the delta.
    """
    from qdrant_finetune.eval.metrics import evaluate_retrieval

    config = config or RFConfig()
    rf_search = RelevanceFeedbackSearch(
        client=client,
        splade_model=splade_model,
        collection_name=collection_name,
        config=config,
    )

    vanilla_results = []
    rf_results = []

    from tqdm import tqdm

    for q in tqdm(queries, desc="Evaluating RF"):
        comparison = rf_search.search_compare(
            query=q["query"],
            limit=max(k_values),
            context_limit=config.context_limit,
        )

        vanilla_ids = [r["payload"].get("product_id", "") for r in comparison["vanilla"]]
        rf_ids = [r["payload"].get("product_id", "") for r in comparison["relevance_feedback"]]

        vanilla_results.append({"query_id": q["query_id"], "retrieved_ids": vanilla_ids})
        rf_results.append({"query_id": q["query_id"], "retrieved_ids": rf_ids})

    vanilla_metrics = evaluate_retrieval(vanilla_results, queries, k_values)
    rf_metrics = evaluate_retrieval(rf_results, queries, k_values)

    delta = {
        f"delta_{k}": rf_metrics[k] - vanilla_metrics[k]
        for k in vanilla_metrics
    }

    return {
        "vanilla": vanilla_metrics,
        "relevance_feedback": rf_metrics,
        "delta": delta,
    }
