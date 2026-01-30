"""ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation) training loop."""

import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset
from rich.console import Console
from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.losses import SpladeLoss, SparseMultipleNegativesRankingLoss
from sentence_transformers.sparse_encoder import SparseEncoderTrainingArguments

from qdrant_finetune.config import FinetuneConfig
from qdrant_finetune.model.sparse_model import create_sparse_encoder, save_sparse_encoder
from qdrant_finetune.qdrant.client import create_sparse_collection, delete_collection, get_qdrant_client
from qdrant_finetune.qdrant.index import index_sparse_vectors
from qdrant_finetune.qdrant.mining import SparseQdrantMiner

logger = logging.getLogger(__name__)
console = Console()


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class ANCELoop:
    """ANCE training loop: train -> index -> mine hard negatives -> retrain."""

    def __init__(
        self,
        config: FinetuneConfig,
        model: Optional[SparseEncoder] = None,
    ):
        self.config = config
        self.model = model or create_sparse_encoder(
            base_model=config.base_model,
            architecture=config.architecture,
            pooling_strategy=config.pooling_strategy,
        )
        self.client = get_qdrant_client(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key or None,
        )

    def run(
        self,
        train_pairs: list[dict],
        products: list[dict],
        eval_callback=None,
    ) -> Path:
        """Run the full ANCE loop.

        Args:
            train_pairs: List of dicts with "query", "positive_text", "positive_ids" keys.
            products: List of product dicts with "product_id" and "text" keys.
            eval_callback: Optional callable(model, iteration) -> metrics dict.

        Returns:
            Path to the final saved model.
        """
        output_dir = Path(self.config.output_dir) / self.config.run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for iteration in range(self.config.ance_iterations):
            console.rule(f"[bold cyan]ANCE Iteration {iteration + 1}/{self.config.ance_iterations}")

            # Build training dataset
            if iteration == 0:
                # First iteration: no hard negatives, just anchor-positive pairs
                dataset = self._build_dataset(train_pairs, with_negatives=False)
            else:
                # Mine hard negatives from previous model
                console.print("[yellow]Mining hard negatives...[/yellow]")
                temp_collection = f"{self.config.collection_name}_ance_{iteration}"

                create_sparse_collection(
                    self.client, temp_collection, recreate=True
                )
                index_sparse_vectors(
                    self.client, temp_collection, self.model, products
                )

                miner = SparseQdrantMiner(
                    client=self.client,
                    model=self.model,
                    collection_name=temp_collection,
                )
                training_examples = miner.mine_for_training(
                    queries=train_pairs,
                    top_k=self.config.mining_top_k,
                    sample_strategy=self.config.sample_strategy,
                    num_negatives=self.config.num_negatives,
                )

                delete_collection(self.client, temp_collection)

                if training_examples:
                    dataset = self._build_dataset(training_examples, with_negatives=True)
                else:
                    logger.warning("No hard negatives mined, reusing pairs without negatives")
                    dataset = self._build_dataset(train_pairs, with_negatives=False)

            # Train
            console.print(f"[green]Training epoch {iteration + 1}...[/green]")
            self._train_epoch(dataset, iteration, output_dir)

            # Eval callback
            if eval_callback:
                metrics = eval_callback(self.model, iteration)
                if metrics:
                    console.print(f"[blue]Metrics: {metrics}[/blue]")

        # Save final model
        final_path = output_dir / "final"
        save_sparse_encoder(self.model, final_path)
        console.print(f"[bold green]Model saved to {final_path}[/bold green]")
        return final_path

    def _build_dataset(self, examples: list[dict], with_negatives: bool) -> Dataset:
        """Build a HuggingFace Dataset from training examples."""
        if with_negatives:
            records = []
            for ex in examples:
                record = {
                    "anchor": ex["anchor"],
                    "positive": ex["positive"],
                }
                # Add negatives as separate columns
                for i, neg in enumerate(ex.get("negative", [])):
                    record[f"negative_{i}"] = neg
                records.append(record)
            return Dataset.from_list(records)
        else:
            return Dataset.from_dict({
                "anchor": [ex.get("query", ex.get("anchor", "")) for ex in examples],
                "positive": [ex.get("positive_text", ex.get("positive", "")) for ex in examples],
            })

    def _train_epoch(self, dataset: Dataset, iteration: int, output_dir: Path):
        """Train the model for one ANCE iteration."""
        # SPLADE sparse tensors don't work on MPS — ensure model is on CPU/CUDA only
        if not _has_cuda():
            self.model.to("cpu")

        inner_loss = SparseMultipleNegativesRankingLoss(model=self.model)
        loss = SpladeLoss(
            model=self.model,
            loss=inner_loss,
            query_regularizer_weight=self.config.query_regularizer_weight,
            document_regularizer_weight=self.config.document_regularizer_weight,
        )

        iter_output = output_dir / f"iteration_{iteration}"

        # Build router mapping for inference_free_splade (Router module)
        router_mapping = {"anchor": "query", "positive": "document"}
        # Add negative columns if present
        col_names = dataset.column_names
        for col in col_names:
            if col.startswith("negative"):
                router_mapping[col] = "document"

        # SPLADE sparse tensors don't work on MPS — force CPU unless CUDA available
        use_cpu = not _has_cuda()

        args = SparseEncoderTrainingArguments(
            output_dir=str(iter_output),
            num_train_epochs=self.config.epochs_per_ance_iteration,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            logging_steps=self.config.logging_steps,
            fp16=_has_cuda(),
            use_cpu=use_cpu,
            router_mapping=router_mapping,
        )

        from sentence_transformers.sparse_encoder import SparseEncoderTrainer

        trainer = SparseEncoderTrainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            loss=loss,
        )
        trainer.train()
