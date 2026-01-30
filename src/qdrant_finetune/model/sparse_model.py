"""Sparse Encoder Model Setup using Sentence Transformers v5."""

import logging
from pathlib import Path
from typing import Literal, Optional, Union

from sentence_transformers import SparseEncoder
from sentence_transformers.models import Router
from sentence_transformers.sparse_encoder.models import (
    MLMTransformer,
    SparseStaticEmbedding,
    SpladePooling,
)

logger = logging.getLogger(__name__)


def _resolve_device(device: Optional[str]) -> str:
    """Pick the best device. SPLADE sparse tensors don't work on MPS, so avoid it."""
    if device is not None:
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    # MPS doesn't support sparse tensor ops needed by SPLADE
    return "cpu"


def create_sparse_encoder(
    base_model: str = "distilbert/distilbert-base-uncased",
    architecture: Literal["splade", "inference_free_splade"] = "inference_free_splade",
    pooling_strategy: Literal["max", "sum"] = "max",
    device: Optional[str] = None,
) -> SparseEncoder:
    """Create a SparseEncoder for fine-tuning."""
    device = _resolve_device(device)

    if "splade" in base_model.lower() and architecture == "splade":
        logger.info(f"Loading pretrained SPLADE model: {base_model}")
        return SparseEncoder(base_model, device=device)

    logger.info(f"Creating {architecture} model from {base_model}")

    if architecture == "splade":
        mlm = MLMTransformer(base_model)
        pooling = SpladePooling(pooling_strategy=pooling_strategy)
        model = SparseEncoder(modules=[mlm, pooling], device=device)

    elif architecture == "inference_free_splade":
        mlm = MLMTransformer(base_model)
        router = Router.for_query_document(
            query_modules=[
                SparseStaticEmbedding(tokenizer=mlm.tokenizer, frozen=False)
            ],
            document_modules=[
                mlm,
                SpladePooling(pooling_strategy=pooling_strategy),
            ],
        )
        model = SparseEncoder(
            modules=[router],
            similarity_fn_name="dot",
            device=device,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


def load_sparse_encoder(
    model_path: Union[str, Path],
    device: Optional[str] = None,
) -> SparseEncoder:
    """Load a trained SparseEncoder from disk."""
    device = _resolve_device(device)
    logger.info(f"Loading SparseEncoder from {model_path}")
    return SparseEncoder(str(model_path), device=device)


def save_sparse_encoder(model: SparseEncoder, output_path: Union[str, Path]) -> Path:
    """Save a SparseEncoder to disk."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    logger.info(f"Saved SparseEncoder to {output_path}")
    return output_path


def get_vocab_size(model: SparseEncoder) -> int:
    """Get vocabulary size of the sparse encoder."""
    for module in model.modules():
        if hasattr(module, "tokenizer"):
            return module.tokenizer.vocab_size
        if hasattr(module, "auto_model") and hasattr(module.auto_model, "config"):
            return module.auto_model.config.vocab_size
    return 30522
