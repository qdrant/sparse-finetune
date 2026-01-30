"""Hard negative mining with Qdrant sparse vector search."""

import logging
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from qdrant_client import QdrantClient
from sentence_transformers import SparseEncoder
from tqdm import tqdm

from qdrant_finetune.qdrant.index import sparse_embedding_to_qdrant

logger = logging.getLogger(__name__)


class SparseQdrantMiner:
    """Mine hard negatives using Qdrant sparse vector search."""

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

    def mine_hard_negatives(
        self,
        queries: List[Dict],
        top_k: int = 20,
        filter_positives: bool = True,
        max_negatives_per_query: int = 5,
        num_workers: int = 8,
    ) -> Dict[str, List[str]]:
        """Mine hard negatives for a batch of queries."""
        logger.info(f"Mining hard negatives for {len(queries)} queries")

        query_texts = [q["query"] for q in queries]
        positive_ids_list = [q.get("positive_ids", set()) for q in queries]

        query_embeddings = self.model.encode(
            query_texts, convert_to_tensor=True, show_progress_bar=True
        )

        hard_negatives: Dict[str, List[str]] = {}

        def search_single(args):
            idx, query_text, emb, positive_ids = args
            try:
                sparse_vector = sparse_embedding_to_qdrant(emb)
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=sparse_vector,
                    using=self.vector_name,
                    limit=top_k,
                    with_payload=True,
                )
                hard_negs = []
                for point in results.points:
                    payload = point.payload or {}
                    product_id = payload.get("product_id", "")
                    product_text = payload.get("text", "")
                    if filter_positives and product_id in positive_ids:
                        continue
                    hard_negs.append(product_text)
                    if len(hard_negs) >= max_negatives_per_query:
                        break
                return query_text, hard_negs
            except Exception as e:
                logger.warning(f"Error mining for query '{query_text[:50]}...': {e}")
                return query_text, []

        args_list = [
            (i, qt, emb, pids)
            for i, (qt, emb, pids) in enumerate(
                zip(query_texts, query_embeddings, positive_ids_list)
            )
        ]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(search_single, args_list),
                    total=len(args_list),
                    desc="Mining hard negatives",
                )
            )

        for query_text, negs in results:
            if negs:
                hard_negatives[query_text] = negs

        logger.info(f"Mined hard negatives for {len(hard_negatives)} queries")
        return hard_negatives

    def mine_for_training(
        self,
        queries: List[Dict],
        top_k: int = 20,
        sample_strategy: str = "top",
        num_negatives: int = 3,
    ) -> List[Dict]:
        """Mine hard negatives and format for training."""
        hard_negatives = self.mine_hard_negatives(
            queries=queries, top_k=top_k, filter_positives=True,
            max_negatives_per_query=top_k,
        )

        training_examples = []
        for query in queries:
            query_text = query["query"]
            positive_text = query.get("positive_text", "")
            if not positive_text or query_text not in hard_negatives:
                continue

            negs = hard_negatives[query_text]
            if len(negs) == 0:
                continue

            n = min(num_negatives, len(negs))
            if sample_strategy == "top":
                selected_negs = negs[:n]
            elif sample_strategy == "mixed":
                n_top = n // 2
                n_mid = n - n_top
                selected_negs = negs[:n_top] + negs[len(negs) // 2 : len(negs) // 2 + n_mid]
            elif sample_strategy == "random":
                selected_negs = random.sample(negs, n)
            else:
                selected_negs = negs[:n]

            training_examples.append({
                "anchor": query_text,
                "positive": positive_text,
                "negative": selected_negs,
            })

        logger.info(f"Created {len(training_examples)} training examples with hard negatives")
        return training_examples
