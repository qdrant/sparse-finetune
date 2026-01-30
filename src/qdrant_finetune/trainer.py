"""Main orchestrator for sparse encoder fine-tuning."""

import logging
from pathlib import Path
from typing import Optional, Union

from rich.console import Console

from qdrant_finetune.ance.loop import ANCELoop
from qdrant_finetune.config import FinetuneConfig, POSITIVE_LABELS
from qdrant_finetune.data.loader import load_products, load_queries
from qdrant_finetune.eval.metrics import QdrantEvaluator, print_metrics
from qdrant_finetune.model.sparse_model import (
    create_sparse_encoder,
    load_sparse_encoder,
    save_sparse_encoder,
)
from qdrant_finetune.qdrant.client import (
    create_sparse_collection,
    get_qdrant_client,
)
from qdrant_finetune.qdrant.index import index_sparse_vectors

logger = logging.getLogger(__name__)
console = Console()


class Trainer:
    """Main entry point for fine-tuning sparse encoders."""

    def __init__(self, config: Union[FinetuneConfig, dict, None] = None, **kwargs):
        if isinstance(config, dict):
            config = FinetuneConfig(**config)
        elif config is None:
            config = FinetuneConfig(**kwargs)
        self.config = config
        self.model = None
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_qdrant_client(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key or None,
            )
        return self._client

    def fit(
        self,
        data: Union[str, list[dict]],
        queries: Optional[Union[str, list[dict]]] = None,
        text_fields: Optional[list[str]] = None,
    ) -> Path:
        """Full fine-tuning pipeline: load data -> (generate queries) -> ANCE train -> save.

        Args:
            data: Path to product data (CSV/JSON/JSONL/HF) or list of product dicts.
            queries: Path to query data or list of query dicts. If None, generates synthetic.
            text_fields: Override auto-detected text fields.

        Returns:
            Path to saved model.
        """
        console.rule("[bold cyan]qdrant-sparse-finetune")

        # Load products
        if isinstance(data, str):
            products = load_products(data, text_fields=text_fields or self.config.text_fields)
        else:
            products = data
        console.print(f"Loaded {len(products)} products")

        # Load or generate queries
        if queries is None:
            console.print("[yellow]No queries provided, generating synthetic queries...[/yellow]")
            from qdrant_finetune.data.synthetic import generate_synthetic_queries
            train_pairs = generate_synthetic_queries(
                products,
                model=self.config.synth_model,
                queries_per_product=self.config.synth_queries_per_product,
            )
        elif isinstance(queries, str):
            query_list = load_queries(queries)
            train_pairs = self._queries_to_pairs(query_list, products)
        else:
            train_pairs = self._queries_to_pairs(queries, products)

        console.print(f"Training with {len(train_pairs)} query-product pairs")

        # Create model
        self.model = create_sparse_encoder(
            base_model=self.config.base_model,
            architecture=self.config.architecture,
            pooling_strategy=self.config.pooling_strategy,
        )

        # Run ANCE loop
        ance = ANCELoop(config=self.config, model=self.model)
        model_path = ance.run(train_pairs=train_pairs, products=products)

        self.model = ance.model
        return model_path

    def evaluate(
        self,
        queries: Union[str, list[dict]],
        collection_name: Optional[str] = None,
        k_values: list[int] = [10, 50, 100],
    ) -> dict:
        """Evaluate the model against a Qdrant collection."""
        if self.model is None:
            raise ValueError("No model loaded. Call fit() or load a model first.")

        if isinstance(queries, str):
            query_list = load_queries(queries)
        else:
            query_list = queries

        collection = collection_name or self.config.collection_name
        evaluator = QdrantEvaluator(
            client=self.client,
            model=self.model,
            collection_name=collection,
        )
        metrics = evaluator.evaluate(query_list, k_values=k_values)
        print_metrics(metrics)
        return metrics

    def index(
        self,
        data: Union[str, list[dict]],
        collection_name: Optional[str] = None,
        text_fields: Optional[list[str]] = None,
    ) -> str:
        """Index products into Qdrant with the current model."""
        if self.model is None:
            raise ValueError("No model loaded. Call fit() or load a model first.")

        if isinstance(data, str):
            products = load_products(data, text_fields=text_fields or self.config.text_fields)
        else:
            products = data

        collection = collection_name or self.config.collection_name
        create_sparse_collection(self.client, collection, recreate=True)
        index_sparse_vectors(self.client, collection, self.model, products)
        console.print(f"[green]Indexed {len(products)} products into '{collection}'[/green]")
        return collection

    def export(self, path: Optional[str] = None) -> Path:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model loaded.")
        out = Path(path) if path else Path(self.config.output_dir) / self.config.run_name / "export"
        return save_sparse_encoder(self.model, out)

    def load(self, path: str, device: Optional[str] = None):
        """Load a previously trained model."""
        self.model = load_sparse_encoder(path, device=device)
        return self

    def _queries_to_pairs(self, query_list: list[dict], products: list[dict]) -> list[dict]:
        """Convert query relevance judgments to anchor-positive training pairs."""
        product_lookup = {p["product_id"]: p["text"] for p in products}
        pairs = []

        for q in query_list:
            # Support synthetic query format (has positive_text + positive_ids directly)
            if "positive_text" in q and "positive_ids" in q:
                pids = set(str(pid) for pid in q["positive_ids"])
                pairs.append({
                    "query": q["query"],
                    "positive_text": q["positive_text"],
                    "positive_ids": pids,
                })
                continue

            # Standard relevance-judgment format
            relevance = q.get("relevance", {})
            positive_ids = set()

            for pid, label in relevance.items():
                if label in POSITIVE_LABELS or (isinstance(label, int) and label >= 1):
                    positive_ids.add(pid)

            for pid in positive_ids:
                if pid in product_lookup:
                    pairs.append({
                        "query": q["query"],
                        "positive_text": product_lookup[pid],
                        "positive_ids": positive_ids,
                    })

        return pairs
