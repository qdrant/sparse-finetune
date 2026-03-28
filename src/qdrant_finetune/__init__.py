"""qdrant-sparse-finetune: Fine-tune SPLADE sparse embeddings with ANCE hard negative mining via Qdrant."""

import os

# MPS (Apple Silicon) doesn't support sparse tensor ops needed by SPLADE.
# Patch MPS availability to False so PyTorch/HF Trainer never moves tensors to MPS.
if not os.environ.get("QDRANT_FINETUNE_ALLOW_MPS"):
    try:
        import torch.backends.mps as _mps
        _mps.is_available = lambda: False
        _mps.is_built = lambda: False
    except Exception:
        pass

from qdrant_finetune.config import FinetuneConfig
from qdrant_finetune.relevance_feedback import (
    RelevanceFeedbackSearch,
    RFConfig,
    RFParams,
)
from qdrant_finetune.trainer import Trainer

__all__ = [
    "Trainer",
    "FinetuneConfig",
    "finetune",
    "RelevanceFeedbackSearch",
    "RFConfig",
    "RFParams",
]


def finetune(
    data: str,
    queries: str | None = None,
    qdrant_url: str = "http://localhost:6333",
    **kwargs,
) -> str:
    """Fine-tune a SPLADE sparse encoder in one call.

    Args:
        data: Path to product data (CSV, JSON, JSONL, or HuggingFace dataset name).
        queries: Path to query data. If None, generates synthetic queries via LLM.
        qdrant_url: Qdrant instance URL.
        **kwargs: Any FinetuneConfig field (ance_iterations, batch_size, synth_model, etc.)

    Returns:
        Path to the saved model.

    Example::

        from qdrant_finetune import finetune

        model_path = finetune("products.csv")

    Or with more control::

        model_path = finetune(
            "products.csv",
            queries="queries.csv",
            qdrant_url="https://my-cluster.qdrant.io",
            qdrant_api_key="...",
            ance_iterations=5,
            synth_model="ollama/llama3",
        )
    """
    config = FinetuneConfig(qdrant_url=qdrant_url, **kwargs)
    trainer = Trainer(config)
    path = trainer.fit(data=data, queries=queries)
    return str(path)
