"""Modal GPU entry point for qdrant-sparse-finetune."""

import modal

app = modal.App("qdrant-sparse-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=5.0",
        "qdrant-client>=1.9",
        "datasets>=2.14",
        "pydantic-settings>=2.0",
        "click>=8.0",
        "pandas>=2.0",
        "tqdm",
        "numpy",
        "torch>=2.0",
        "pyyaml",
        "litellm>=1.0",
        "rich>=13.0",
        "accelerate>=1.1.0",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src/qdrant_finetune", remote_path="/root/src/qdrant_finetune", copy=True)
)

# Mount for local data files
data_volume = modal.Volume.from_name("finetune-data", create_if_missing=True)
output_volume = modal.Volume.from_name("finetune-output", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/data": data_volume, "/output": output_volume},
    secrets=[
        modal.Secret.from_name("qdrant-secret", required_keys=["QDRANT_URL", "QDRANT_API_KEY"]),
    ],
)
def train(
    data_path: str = "/data/products.csv",
    queries_path: str | None = None,
    config_override: dict | None = None,
):
    """Run training on Modal GPU."""
    import os

    from qdrant_finetune import FinetuneConfig, Trainer

    config_kwargs = {
        "qdrant_url": os.environ["QDRANT_URL"],
        "qdrant_api_key": os.environ["QDRANT_API_KEY"],
        "output_dir": "/output",
        "run_name": "modal_run",
    }
    if config_override:
        config_kwargs.update(config_override)

    config = FinetuneConfig(**config_kwargs)
    trainer = Trainer(config=config)
    model_path = trainer.fit(data=data_path, queries=queries_path)

    # Index final model (may fail if qdrant-client version mismatch)
    try:
        trainer.index(data=data_path)
    except Exception as e:
        print(f"Warning: Post-training indexing failed: {e}")

    # Evaluate if queries provided
    result = {"model_path": str(model_path)}
    if queries_path:
        try:
            metrics = trainer.evaluate(queries=queries_path)
            result["metrics"] = metrics
        except Exception as e:
            print(f"Warning: Evaluation failed: {e}")

    output_volume.commit()
    return result


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/data": data_volume, "/output": output_volume},
    secrets=[
        modal.Secret.from_name("qdrant-secret", required_keys=["QDRANT_URL", "QDRANT_API_KEY"]),
    ],
)
def evaluate(
    model_path: str = "/output/modal_run/final",
    queries_path: str = "/data/queries.csv",
    collection_name: str | None = None,
):
    """Evaluate a trained model on Modal GPU."""
    import os

    from qdrant_finetune import FinetuneConfig, Trainer

    config = FinetuneConfig(
        qdrant_url=os.environ["QDRANT_URL"],
        qdrant_api_key=os.environ["QDRANT_API_KEY"],
    )
    trainer = Trainer(config=config)
    trainer.load(model_path)
    return trainer.evaluate(queries=queries_path, collection_name=collection_name)


@app.function(
    image=image,
    timeout=1800,
    volumes={"/data": data_volume},
)
def generate_queries(
    data_path: str = "/data/products.csv",
    output_path: str = "/data/queries.jsonl",
    synth_model: str = "gpt-4o-mini",
    queries_per_product: int = 3,
    max_products: int | None = None,
):
    """Generate synthetic queries on Modal (no GPU needed)."""
    import json

    from qdrant_finetune.data.loader import load_products
    from qdrant_finetune.data.synthetic import generate_synthetic_queries

    products = load_products(data_path)
    pairs = generate_synthetic_queries(
        products,
        model=synth_model,
        queries_per_product=queries_per_product,
        max_products=max_products,
    )

    with open(output_path, "w") as f:
        for pair in pairs:
            record = {k: list(v) if isinstance(v, set) else v for k, v in pair.items()}
            f.write(json.dumps(record) + "\n")

    data_volume.commit()
    return {"output_path": output_path, "num_pairs": len(pairs)}


@app.local_entrypoint()
def main(
    data: str = "/data/products.csv",
    queries: str | None = None,
    synth_model: str = "gpt-4o-mini",
    ance_iterations: int = 3,
    batch_size: int = 32,
):
    """Run full pipeline: optionally generate queries, then train."""
    config_override = {
        "synth_model": synth_model,
        "ance_iterations": ance_iterations,
        "batch_size": batch_size,
    }

    result = train.remote(
        data_path=data,
        queries_path=queries,
        config_override=config_override,
    )
    print(f"Result: {result}")
