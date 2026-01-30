"""Configuration for sparse encoder fine-tuning."""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FinetuneConfig(BaseSettings):
    """Fine-tuning configuration. Reads from .env, YAML, or kwargs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="QDRANT_FINETUNE_",
        extra="ignore",
    )

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant URL")
    qdrant_api_key: str = Field(default="", description="Qdrant API key")

    # Model
    base_model: str = Field(
        default="distilbert/distilbert-base-uncased",
        description="Base HuggingFace model",
    )
    architecture: Literal["splade", "inference_free_splade"] = Field(
        default="inference_free_splade"
    )
    pooling_strategy: Literal["max", "sum"] = Field(default="max")

    # Data
    text_fields: Optional[list[str]] = Field(
        default=None,
        description="Column names to use as product text. Auto-detected if None.",
    )

    # Training
    batch_size: int = Field(default=32)
    learning_rate: float = Field(default=2e-5)
    num_epochs: int = Field(default=3)
    warmup_ratio: float = Field(default=0.1)
    query_regularizer_weight: float = Field(default=5e-5)
    document_regularizer_weight: float = Field(default=3e-5)

    # ANCE
    ance_iterations: int = Field(default=3)
    mining_top_k: int = Field(default=20)
    num_negatives: int = Field(default=3)
    epochs_per_ance_iteration: int = Field(default=1)
    sample_strategy: Literal["top", "mixed", "random"] = Field(default="top")

    # Synthetic query generation
    synth_model: str = Field(
        default="gpt-4o-mini",
        description="litellm model string for synthetic query generation",
    )
    synth_queries_per_product: int = Field(default=3)

    # Output
    output_dir: Path = Field(default=Path("./output"))
    run_name: str = Field(default="finetune")
    collection_name: str = Field(default="sparse_finetune")

    # Checkpointing
    save_steps: int = Field(default=1000)
    save_total_limit: int = Field(default=3)
    logging_steps: int = Field(default=100)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "FinetuneConfig":
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        config_dict.update(overrides)
        return cls(**config_dict)


# Label scoring systems
ESCI_SCORES = {
    "E": 1.0, "Exact": 1.0,
    "S": 0.7, "Substitute": 0.7,
    "C": 0.5, "Complement": 0.5,
    "I": 0.0, "Irrelevant": 0.0,
}

POSITIVE_LABELS = {"E", "Exact", "S", "Substitute"}
RELEVANT_LABELS = {"E", "Exact", "S", "Substitute"}

WANDS_SCORES = {2: 1.0, 1: 0.5, 0: 0.0}
WANDS_RELEVANT_LABELS = {2, 1}
