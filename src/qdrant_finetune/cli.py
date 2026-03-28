"""CLI interface for qdrant-sparse-finetune."""

import logging
import os
from pathlib import Path

# MPS (Apple Silicon) doesn't support sparse tensor ops needed by SPLADE.
# Patch MPS availability to False so PyTorch/HF Trainer never moves tensors to MPS.
if not os.environ.get("QDRANT_FINETUNE_ALLOW_MPS"):
    try:
        import torch.backends.mps as _mps
        _mps.is_available = lambda: False
        _mps.is_built = lambda: False
    except Exception:
        pass

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Confirm, Prompt

console = Console()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def cli(verbose):
    """qdrant-sparse-finetune: Fine-tune SPLADE sparse embeddings with ANCE via Qdrant."""
    setup_logging(verbose)


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

@cli.command()
def setup():
    """Interactive setup wizard — configure Qdrant, LLM provider, and optionally Modal."""
    console.print("\n[bold cyan]qdrant-sparse-finetune setup[/bold cyan]\n")

    env_lines: list[str] = []

    # --- Qdrant ---
    console.print("[bold]1. Qdrant connection[/bold]")
    qdrant_url = Prompt.ask(
        "  Qdrant URL",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
    )
    qdrant_api_key = Prompt.ask(
        "  Qdrant API key (blank for local/no auth)",
        default=os.environ.get("QDRANT_API_KEY", ""),
    )
    env_lines.append(f"QDRANT_URL={qdrant_url}")
    if qdrant_api_key:
        env_lines.append(f"QDRANT_API_KEY={qdrant_api_key}")

    # Validate connection
    console.print("  Testing connection...", end=" ")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=10)
        collections = client.get_collections()
        console.print(f"[green]✓[/green] Connected ({len(collections.collections)} collections)")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed: {e}")
        if not Confirm.ask("  Continue anyway?", default=False):
            return

    # --- LLM provider ---
    console.print("\n[bold]2. LLM provider (for synthetic query generation)[/bold]")
    console.print("  Used when you don't have labeled queries.")
    provider = Prompt.ask(
        "  Provider",
        choices=["openai", "anthropic", "openrouter", "ollama", "skip"],
        default="openai",
    )

    llm_env: dict[str, str] = {}
    synth_model = "gpt-4o-mini"

    if provider == "openai":
        key = Prompt.ask("  OpenAI API key", default=os.environ.get("OPENAI_API_KEY", ""))
        if key:
            env_lines.append(f"OPENAI_API_KEY={key}")
            llm_env["OPENAI_API_KEY"] = key
        synth_model = Prompt.ask("  Model", default="gpt-4o-mini")
    elif provider == "anthropic":
        key = Prompt.ask("  Anthropic API key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
        if key:
            env_lines.append(f"ANTHROPIC_API_KEY={key}")
            llm_env["ANTHROPIC_API_KEY"] = key
        synth_model = Prompt.ask("  Model", default="anthropic/claude-sonnet-4-20250514")
    elif provider == "openrouter":
        key = Prompt.ask("  OpenRouter API key", default=os.environ.get("OPENROUTER_API_KEY", ""))
        if key:
            env_lines.append(f"OPENROUTER_API_KEY={key}")
            llm_env["OPENROUTER_API_KEY"] = key
        synth_model = Prompt.ask("  Model", default="openrouter/google/gemma-2-9b-it")
    elif provider == "ollama":
        console.print("  [dim]No API key needed. Make sure Ollama is running locally.[/dim]")
        synth_model = Prompt.ask("  Model", default="ollama/llama3")
    else:
        console.print("  [dim]Skipped. You'll need labeled queries for training.[/dim]")

    env_lines.append(f"QDRANT_FINETUNE_SYNTH_MODEL={synth_model}")

    # --- GPU provider ---
    console.print("\n[bold]3. Cloud GPU provider[/bold]")
    gpu_provider = Prompt.ask(
        "  Provider",
        choices=["modal", "vultr", "skip"],
        default="skip",
    )

    use_modal = False
    if gpu_provider == "modal":
        use_modal = True
        try:
            from qdrant_finetune.modal_runner import setup_modal
            setup_modal(
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                llm_env=llm_env if llm_env else None,
            )
        except Exception as e:
            console.print(f"  [red]Modal setup failed: {e}[/red]")
            console.print("  You can set up Modal later with: modal setup")
    elif gpu_provider == "vultr":
        vultr_key = Prompt.ask(
            "  Vultr API key",
            default=os.environ.get("VULTR_API_KEY", ""),
        )
        if vultr_key:
            env_lines.append(f"VULTR_API_KEY={vultr_key}")
            console.print("  [green]✓[/green] Vultr API key saved")
        console.print("  [yellow]Vultr GPU training coming soon. Key saved for when it's ready.[/yellow]")
    else:
        console.print("  [dim]Skipped. You can train locally or add a provider later.[/dim]")

    # --- Write .env ---
    env_path = Path(".env")
    if env_path.exists():
        if not Confirm.ask(f"\n  .env already exists. Overwrite?", default=False):
            console.print("  [dim]Keeping existing .env[/dim]")
            console.print("\n[bold green]Setup complete![/bold green]")
            return

    env_path.write_text("\n".join(env_lines) + "\n")
    console.print(f"\n  [green]✓[/green] Wrote {env_path}")

    console.print("\n[bold green]Setup complete![/bold green]")
    console.print("\nNext steps:")
    console.print("  qdrant-finetune train --data products.csv")
    if use_modal:
        console.print("  qdrant-finetune train --data products.csv --gpu modal")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--data", required=True, help="Path to product data (CSV/JSON/JSONL/HF dataset)")
@click.option("--queries", default=None, help="Path to query data with relevance labels")
@click.option("--config", "config_path", default=None, help="Path to YAML config file")
@click.option("--qdrant-url", default=None, help="Qdrant URL")
@click.option("--qdrant-api-key", default=None, help="Qdrant API key")
@click.option("--base-model", default=None, help="Base HuggingFace model")
@click.option("--output-dir", default=None, help="Output directory")
@click.option("--ance-iterations", default=None, type=int, help="Number of ANCE iterations")
@click.option("--batch-size", default=None, type=int, help="Training batch size")
@click.option("--synth-model", default=None, help="LLM for synthetic query generation")
@click.option(
    "--gpu",
    required=True,
    type=click.Choice(["modal", "vultr"]),
    help="GPU backend: modal or vultr",
)
def train(data, queries, config_path, gpu, **kwargs):
    """Train a SPLADE sparse encoder with ANCE hard negative mining."""
    from qdrant_finetune.job_store import JobStore
    store = JobStore()

    overrides = {k: v for k, v in kwargs.items() if v is not None}
    overrides = {k.replace("-", "_"): v for k, v in overrides.items()}

    job = store.create_job("train", config={"data": data, "queries": queries, "gpu": gpu, **overrides}, source="cli")
    job_id = job["id"]
    store.update_job(job_id, status="running", started_at=__import__("datetime").datetime.utcnow().isoformat())
    store.append_log(job_id, f"Starting training via CLI (gpu={gpu})...")

    try:
        if gpu == "modal":
            _train_modal(data, queries, overrides)
        elif gpu == "vultr":
            _train_vultr(data, queries, overrides)

        store.update_job(job_id, status="completed", completed_at=__import__("datetime").datetime.utcnow().isoformat())
        store.append_log(job_id, "Training completed.")
    except Exception as e:
        store.update_job(job_id, status="failed", error=str(e), completed_at=__import__("datetime").datetime.utcnow().isoformat())
        store.append_log(job_id, f"ERROR: {e}")
        raise


def _train_modal(data, queries, overrides):
    """Upload data and train on Modal GPU."""
    from qdrant_finetune.modal_runner import run_on_modal

    run_on_modal(
        data_path=data,
        queries_path=queries,
        config_override=overrides if overrides else None,
    )


def _train_vultr(data, queries, overrides):
    """Upload data and train on Vultr GPU."""
    from qdrant_finetune.vultr_runner import run_on_vultr

    run_on_vultr(
        data_path=data,
        queries_path=queries,
        config_override=overrides if overrides else None,
    )


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", required=True, help="Path to trained model")
@click.option("--queries", required=True, help="Path to query data with relevance labels")
@click.option("--collection", default=None, help="Qdrant collection name")
@click.option("--qdrant-url", default=None, help="Qdrant URL")
@click.option("--qdrant-api-key", default=None, help="Qdrant API key")
def evaluate(model, queries, collection, **kwargs):
    """Evaluate a trained model against a Qdrant collection."""
    from datetime import datetime

    from qdrant_finetune.config import FinetuneConfig
    from qdrant_finetune.job_store import JobStore
    from qdrant_finetune.trainer import Trainer

    store = JobStore()
    job = store.create_job("evaluate", config={"model": model, "queries": queries, "collection": collection}, source="cli")
    job_id = job["id"]
    store.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())

    try:
        overrides = {k.replace("-", "_"): v for k, v in kwargs.items() if v is not None}
        config = FinetuneConfig(**overrides)
        trainer = Trainer(config=config)
        trainer.load(model)
        metrics = trainer.evaluate(queries=queries, collection_name=collection)
        store.update_job(job_id, status="completed", results=metrics, completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, f"Evaluation completed: {metrics}")
    except Exception as e:
        store.update_job(job_id, status="failed", error=str(e), completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, f"ERROR: {e}")
        raise


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", required=True, help="Path to trained model")
@click.option("--data", required=True, help="Path to product data")
@click.option("--collection", default=None, help="Qdrant collection name")
@click.option("--qdrant-url", default=None, help="Qdrant URL")
@click.option("--qdrant-api-key", default=None, help="Qdrant API key")
def index(model, data, collection, **kwargs):
    """Index products into Qdrant with a trained model."""
    from datetime import datetime

    from qdrant_finetune.config import FinetuneConfig
    from qdrant_finetune.job_store import JobStore
    from qdrant_finetune.trainer import Trainer

    store = JobStore()
    job = store.create_job("index", config={"model": model, "data": data, "collection": collection}, source="cli")
    job_id = job["id"]
    store.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())

    try:
        overrides = {k.replace("-", "_"): v for k, v in kwargs.items() if v is not None}
        config = FinetuneConfig(**overrides)
        trainer = Trainer(config=config)
        trainer.load(model)
        trainer.index(data=data, collection_name=collection)
        store.update_job(job_id, status="completed", completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, "Indexing completed.")
    except Exception as e:
        store.update_job(job_id, status="failed", error=str(e), completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, f"ERROR: {e}")
        raise


# ---------------------------------------------------------------------------
# generate-queries
# ---------------------------------------------------------------------------

@cli.command("generate-queries")
@click.option("--data", required=True, help="Path to product data")
@click.option("--output", default="queries.jsonl", help="Output file path")
@click.option("--synth-model", default="gpt-4o-mini", help="LLM model for generation")
@click.option("--queries-per-product", default=3, type=int, help="Queries per product")
@click.option("--max-products", default=None, type=int, help="Limit products to process")
def generate_queries(data, output, synth_model, queries_per_product, max_products):
    """Generate synthetic search queries from product data."""
    import json
    from datetime import datetime

    from qdrant_finetune.data.loader import load_products
    from qdrant_finetune.data.synthetic import generate_synthetic_queries
    from qdrant_finetune.job_store import JobStore

    store = JobStore()
    job = store.create_job("generate-queries", config={"data": data, "output": output, "model": synth_model}, source="cli")
    job_id = job["id"]
    store.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())

    try:
        products = load_products(data)
        pairs = generate_synthetic_queries(
            products,
            model=synth_model,
            queries_per_product=queries_per_product,
            max_products=max_products,
        )

        with open(output, "w") as f:
            for pair in pairs:
                record = {k: list(v) if isinstance(v, set) else v for k, v in pair.items()}
                f.write(json.dumps(record) + "\n")

        console.print(f"[green]Wrote {len(pairs)} query-product pairs to {output}[/green]")
        results_data = [
            {"query": p.get("query", ""), "product_id": p.get("product_id", ""), "positive_text": p.get("positive_text", "")}
            for p in pairs
        ]
        store.update_job(job_id, status="completed", results={"output_file": output, "count": len(pairs), "queries": results_data}, completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, f"Generated {len(pairs)} pairs to {output}")
    except Exception as e:
        store.update_job(job_id, status="failed", error=str(e), completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, f"ERROR: {e}")
        raise


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", required=True, help="Path to trained model on disk")
@click.option("--repo", required=True, help="HuggingFace repo name (e.g. my-org/my-splade-model)")
@click.option("--hf-token", default=None, help="HuggingFace token (defaults to HF_TOKEN env var)")
@click.option("--private", is_flag=True, default=False, help="Create a private repo")
def publish(model, repo, hf_token, private):
    """Publish a trained sparse encoder to HuggingFace Hub."""
    from datetime import datetime

    from qdrant_finetune.job_store import JobStore

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        console.print("[red]No HuggingFace token provided. Set HF_TOKEN or use --hf-token.[/red]")
        raise SystemExit(1)

    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model path does not exist: {model_path}[/red]")
        raise SystemExit(1)

    store = JobStore()
    job = store.create_job("publish", config={"model": model, "repo": repo, "private": private}, source="cli")
    job_id = job["id"]
    store.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())

    try:
        console.print(f"Loading model from [cyan]{model_path}[/cyan]...")
        from sentence_transformers import SparseEncoder
        encoder = SparseEncoder(str(model_path))

        console.print(f"Pushing to [cyan]{repo}[/cyan] (private={private})...")
        encoder.push_to_hub(repo, token=token, private=private)

        console.print(f"[green]Published to https://huggingface.co/{repo}[/green]")
        store.update_job(job_id, status="completed", completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, f"Published to https://huggingface.co/{repo}")
    except Exception as e:
        store.update_job(job_id, status="failed", error=str(e), completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, f"ERROR: {e}")
        raise


# ---------------------------------------------------------------------------
# studio (dashboard)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--port", default=7777, type=int, help="Port to run the dashboard on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
def studio(port, host):
    """Launch the web dashboard for training, evaluation, and publishing."""
    import subprocess
    import sys

    dashboard_path = Path(__file__).parent.parent.parent / "dashboard" / "server.py"
    if not dashboard_path.exists():
        # Try installed package location
        dashboard_path = Path(__file__).parent / ".." / ".." / "dashboard" / "server.py"
        dashboard_path = dashboard_path.resolve()

    if not dashboard_path.exists():
        console.print("[red]Dashboard not found. Ensure the dashboard/ directory is present.[/red]")
        raise SystemExit(1)

    console.print(f"[bold cyan]Launching studio at http://{host}:{port}[/bold cyan]")
    subprocess.run([sys.executable, str(dashboard_path), "--port", str(port), "--host", host])


# ---------------------------------------------------------------------------
# pipeline (end-to-end)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--data", required=True, help="Path to product data (CSV/JSON/JSONL/HF dataset)")
@click.option("--queries", default=None, help="Path to existing queries (skip generation if provided)")
@click.option("--gpu", required=True, type=click.Choice(["modal", "vultr"]), help="GPU backend")
@click.option("--repo", default=None, help="HuggingFace repo to publish to (skip publish if omitted)")
@click.option("--hf-token", default=None, help="HuggingFace token (defaults to HF_TOKEN env var)")
@click.option("--private", is_flag=True, default=False, help="Private HF repo")
@click.option("--synth-model", default="gpt-4o-mini", help="LLM for query generation")
@click.option("--queries-per-product", default=3, type=int, help="Queries per product")
@click.option("--max-products", default=None, type=int, help="Limit products for query generation")
@click.option("--base-model", default=None, help="Base HuggingFace model")
@click.option("--ance-iterations", default=None, type=int, help="ANCE iterations")
@click.option("--batch-size", default=None, type=int, help="Training batch size")
@click.option("--run-name", default="finetune", help="Run name for output directory")
def pipeline(data, queries, gpu, repo, hf_token, private, synth_model,
             queries_per_product, max_products, base_model, ance_iterations,
             batch_size, run_name):
    """End-to-end pipeline: generate queries → train → evaluate → publish."""
    import json
    from datetime import datetime

    from qdrant_finetune.job_store import JobStore

    store = JobStore()
    job = store.create_job("pipeline", config={
        "data": data, "gpu": gpu, "repo": repo, "run_name": run_name,
    }, source="cli")
    job_id = job["id"]
    store.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())

    queries_path = queries
    output_dir = Path("output") / run_name
    model_path = output_dir / "final"

    try:
        # --- Step 1: Generate queries (if not provided) ---
        if not queries_path:
            console.print("\n[bold cyan]Step 1/4: Generating synthetic queries...[/bold cyan]")
            store.append_log(job_id, "Step 1: Generating synthetic queries...")

            from qdrant_finetune.data.loader import load_products
            from qdrant_finetune.data.synthetic import generate_synthetic_queries

            products = load_products(data)
            pairs = generate_synthetic_queries(
                products,
                model=synth_model,
                queries_per_product=queries_per_product,
                max_products=max_products,
            )

            queries_path = "pipeline_queries.jsonl"
            with open(queries_path, "w") as f:
                for pair in pairs:
                    record = {k: list(v) if isinstance(v, set) else v for k, v in pair.items()}
                    f.write(json.dumps(record) + "\n")

            console.print(f"  [green]✓[/green] Generated {len(pairs)} query-product pairs")
            store.append_log(job_id, f"Generated {len(pairs)} queries → {queries_path}")
        else:
            console.print("\n[bold cyan]Step 1/4: Using provided queries[/bold cyan]")
            store.append_log(job_id, f"Step 1: Using provided queries from {queries_path}")

        # --- Step 2: Train ---
        console.print(f"\n[bold cyan]Step 2/4: Training on {gpu}...[/bold cyan]")
        store.append_log(job_id, f"Step 2: Training on {gpu}...")

        overrides = {k: v for k, v in {
            "base_model": base_model,
            "ance_iterations": ance_iterations,
            "batch_size": batch_size,
            "run_name": run_name,
        }.items() if v is not None}

        if gpu == "modal":
            from qdrant_finetune.modal_runner import run_on_modal
            run_on_modal(data_path=data, queries_path=queries_path, config_override=overrides or None)
        elif gpu == "vultr":
            from qdrant_finetune.vultr_runner import run_on_vultr
            run_on_vultr(data_path=data, queries_path=queries_path, config_override=overrides or None)

        console.print(f"  [green]✓[/green] Training completed → {model_path}")
        store.append_log(job_id, f"Training completed → {model_path}")

        # --- Step 3: Evaluate ---
        console.print(f"\n[bold cyan]Step 3/4: Evaluating...[/bold cyan]")
        store.append_log(job_id, "Step 3: Evaluating...")

        if model_path.exists():
            from qdrant_finetune.config import FinetuneConfig
            from qdrant_finetune.trainer import Trainer

            config = FinetuneConfig()
            trainer = Trainer(config=config)
            trainer.load(str(model_path))
            metrics = trainer.evaluate(queries=queries_path)
            console.print(f"  [green]✓[/green] Metrics: {metrics}")
            store.append_log(job_id, f"Evaluation metrics: {metrics}")
            store.update_job(job_id, results={"metrics": metrics})
        else:
            console.print(f"  [yellow]⚠[/yellow] Model path not found locally (may be on remote GPU). Skipping eval.")
            store.append_log(job_id, "Skipped evaluation — model not found locally")

        # --- Step 4: Publish ---
        if not repo and model_path.exists():
            if click.confirm("\nTraining complete! Publish to HuggingFace Hub?", default=True):
                repo = click.prompt("  Repo name (e.g. your-username/my-splade-model)")
                private = click.confirm("  Private repo?", default=False)

        if repo:
            console.print(f"\n[bold cyan]Step 4/4: Publishing to {repo}...[/bold cyan]")
            store.append_log(job_id, f"Step 4: Publishing to {repo}...")

            token = hf_token or os.environ.get("HF_TOKEN")
            if not token:
                token = click.prompt("  HF_TOKEN not set. Enter your HuggingFace token", hide_input=True)

            if not model_path.exists():
                console.print("  [yellow]⚠[/yellow] Model not found locally — skipping publish.")
                store.append_log(job_id, "Skipped publish — model not found locally")
            else:
                from sentence_transformers import SparseEncoder
                encoder = SparseEncoder(str(model_path))
                encoder.push_to_hub(repo, token=token, private=private)
                hf_url = f"https://huggingface.co/{repo}"
                console.print(f"\n[bold green]Published → {hf_url}[/bold green]")
                store.append_log(job_id, f"Published → {hf_url}")
        else:
            console.print("\n[dim]Step 4/4: Publish skipped[/dim]")
            store.append_log(job_id, "Step 4: Skipped publish")

        # --- Done ---
        console.print("\n[bold green]Pipeline complete![/bold green]")
        store.update_job(job_id, status="completed", completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, "Pipeline completed successfully.")

    except Exception as e:
        store.update_job(job_id, status="failed", error=str(e), completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, f"ERROR: {e}")
        console.print(f"\n[red]Pipeline failed: {e}[/red]")
        raise


# ---------------------------------------------------------------------------
# relevance-feedback (train params)
# ---------------------------------------------------------------------------

@cli.command("train-rf")
@click.option("--model", required=True, help="Path to trained SPLADE model")
@click.option("--queries", required=True, help="Path to query data (one query per line, or JSONL with 'query' field)")
@click.option("--collection", default=None, help="Qdrant collection name")
@click.option("--feedback-model", default="mixedbread-ai/mxbai-embed-large-v1", help="Feedback embedding model")
@click.option("--context-limit", default=10, type=int, help="Number of initial results for feedback")
@click.option("--qdrant-url", default=None, help="Qdrant URL")
@click.option("--qdrant-api-key", default=None, help="Qdrant API key")
@click.option("--output", default="rf_params.json", help="Output file for learned parameters")
def train_rf(model, queries, collection, feedback_model, context_limit, output, **kwargs):
    """Train relevance feedback parameters (a, b, c) for your SPLADE model and collection.

    Requires: pip install qdrant-sparse-finetune[rf]
    """
    import json
    from datetime import datetime

    from qdrant_finetune.config import FinetuneConfig
    from qdrant_finetune.job_store import JobStore
    from qdrant_finetune.relevance_feedback import RFConfig, RFParams, train_rf_params

    store = JobStore()
    job = store.create_job("train-rf", config={
        "model": model, "queries": queries, "feedback_model": feedback_model,
    }, source="cli")
    job_id = job["id"]
    store.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())

    try:
        # load SPLADE model
        from qdrant_finetune.model.sparse_model import load_sparse_encoder
        splade = load_sparse_encoder(model)

        # load queries
        query_texts = _load_query_texts(queries)
        console.print(f"Loaded {len(query_texts)} queries for RF training")

        # qdrant client
        overrides = {k.replace("-", "_"): v for k, v in kwargs.items() if v is not None}
        config = FinetuneConfig(**overrides)
        from qdrant_finetune.qdrant.client import get_qdrant_client
        client = get_qdrant_client(url=config.qdrant_url, api_key=config.qdrant_api_key or None)

        coll = collection or config.collection_name

        rf_config = RFConfig(
            feedback_model=feedback_model,
            context_limit=context_limit,
        )

        console.print(f"Training RF params with feedback model: [cyan]{feedback_model}[/cyan]")
        params = train_rf_params(
            client=client,
            splade_model=splade,
            collection_name=coll,
            queries=query_texts,
            config=rf_config,
        )

        # save params
        out_path = Path(output)
        out_path.write_text(json.dumps(params.to_dict(), indent=2) + "\n")
        console.print(f"\n[green]Learned RF params:[/green] a={params.a:.4f}, b={params.b:.4f}, c={params.c:.4f}")
        console.print(f"[green]Saved to {out_path}[/green]")

        store.update_job(job_id, status="completed", results=params.to_dict(), completed_at=datetime.utcnow().isoformat())
    except Exception as e:
        store.update_job(job_id, status="failed", error=str(e), completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, f"ERROR: {e}")
        raise


# ---------------------------------------------------------------------------
# relevance-feedback (evaluate)
# ---------------------------------------------------------------------------

@cli.command("eval-rf")
@click.option("--model", required=True, help="Path to trained SPLADE model")
@click.option("--queries", required=True, help="Path to query data with relevance labels")
@click.option("--collection", default=None, help="Qdrant collection name")
@click.option("--feedback-model", default="mixedbread-ai/mxbai-embed-large-v1", help="Feedback embedding model")
@click.option("--context-limit", default=10, type=int, help="Number of initial results for feedback")
@click.option("--params", "params_file", default=None, help="Path to RF params JSON (from train-rf)")
@click.option("--qdrant-url", default=None, help="Qdrant URL")
@click.option("--qdrant-api-key", default=None, help="Qdrant API key")
def eval_rf(model, queries, collection, feedback_model, context_limit, params_file, **kwargs):
    """Evaluate relevance feedback vs vanilla SPLADE retrieval side-by-side."""
    import json
    from datetime import datetime

    from qdrant_finetune.config import FinetuneConfig
    from qdrant_finetune.data.loader import load_queries as load_query_data
    from qdrant_finetune.eval.metrics import print_metrics
    from qdrant_finetune.job_store import JobStore
    from qdrant_finetune.relevance_feedback import RFConfig, RFParams, evaluate_rf

    store = JobStore()
    job = store.create_job("eval-rf", config={
        "model": model, "queries": queries, "feedback_model": feedback_model,
    }, source="cli")
    job_id = job["id"]
    store.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())

    try:
        from qdrant_finetune.model.sparse_model import load_sparse_encoder
        splade = load_sparse_encoder(model)

        query_list = load_query_data(queries)
        console.print(f"Loaded {len(query_list)} queries for evaluation")

        overrides = {k.replace("-", "_"): v for k, v in kwargs.items() if v is not None}
        config = FinetuneConfig(**overrides)
        from qdrant_finetune.qdrant.client import get_qdrant_client
        client = get_qdrant_client(url=config.qdrant_url, api_key=config.qdrant_api_key or None)

        coll = collection or config.collection_name

        params = RFParams()
        if params_file:
            with open(params_file) as f:
                params = RFParams.from_dict(json.load(f))
            console.print(f"Using RF params from {params_file}: a={params.a:.4f}, b={params.b:.4f}, c={params.c:.4f}")

        rf_config = RFConfig(
            feedback_model=feedback_model,
            context_limit=context_limit,
            params=params,
        )

        results = evaluate_rf(
            client=client,
            splade_model=splade,
            collection_name=coll,
            queries=query_list,
            config=rf_config,
        )

        console.print()
        print_metrics(results["vanilla"], title="Vanilla SPLADE")
        print_metrics(results["relevance_feedback"], title="SPLADE + Relevance Feedback")
        print_metrics(results["delta"], title="Delta (RF - Vanilla)")

        store.update_job(job_id, status="completed", results=results, completed_at=datetime.utcnow().isoformat())
    except Exception as e:
        store.update_job(job_id, status="failed", error=str(e), completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, f"ERROR: {e}")
        raise


# ---------------------------------------------------------------------------
# relevance-feedback (interactive query)
# ---------------------------------------------------------------------------

@cli.command("query-rf")
@click.option("--model", required=True, help="Path to trained SPLADE model")
@click.option("--collection", default=None, help="Qdrant collection name")
@click.option("--feedback-model", default="mixedbread-ai/mxbai-embed-large-v1", help="Feedback embedding model")
@click.option("--context-limit", default=10, type=int, help="Number of initial results for feedback")
@click.option("--limit", default=10, type=int, help="Number of final results")
@click.option("--params", "params_file", default=None, help="Path to RF params JSON")
@click.option("--qdrant-url", default=None, help="Qdrant URL")
@click.option("--qdrant-api-key", default=None, help="Qdrant API key")
@click.argument("query")
def query_rf(model, collection, feedback_model, context_limit, limit, params_file, query, **kwargs):
    """Run a single query with relevance feedback and compare to vanilla results."""
    import json

    from rich.table import Table

    from qdrant_finetune.config import FinetuneConfig
    from qdrant_finetune.model.sparse_model import load_sparse_encoder
    from qdrant_finetune.qdrant.client import get_qdrant_client
    from qdrant_finetune.relevance_feedback import RFConfig, RFParams, RelevanceFeedbackSearch

    splade = load_sparse_encoder(model)

    overrides = {k.replace("-", "_"): v for k, v in kwargs.items() if v is not None}
    config = FinetuneConfig(**overrides)
    client = get_qdrant_client(url=config.qdrant_url, api_key=config.qdrant_api_key or None)

    coll = collection or config.collection_name

    params = RFParams()
    if params_file:
        with open(params_file) as f:
            params = RFParams.from_dict(json.load(f))

    rf_config = RFConfig(
        feedback_model=feedback_model,
        context_limit=context_limit,
        rf_limit=limit,
        params=params,
    )

    rf_search = RelevanceFeedbackSearch(
        client=client,
        splade_model=splade,
        collection_name=coll,
        config=rf_config,
    )

    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print(f"[dim]Feedback model: {feedback_model} | context_limit: {context_limit}[/dim]\n")

    comparison = rf_search.search_compare(query, limit=limit, context_limit=context_limit)

    # vanilla results
    table = Table(title="Vanilla SPLADE")
    table.add_column("#", style="dim")
    table.add_column("Score", style="cyan")
    table.add_column("Product", style="white")
    for i, r in enumerate(comparison["vanilla"][:limit], 1):
        text = r["payload"].get("text", "")[:100]
        table.add_row(str(i), f"{r['score']:.4f}", text)
    console.print(table)

    console.print()

    # RF results
    table = Table(title="SPLADE + Relevance Feedback")
    table.add_column("#", style="dim")
    table.add_column("Score", style="green")
    table.add_column("Product", style="white")
    for i, r in enumerate(comparison["relevance_feedback"][:limit], 1):
        text = r["payload"].get("text", "")[:100]
        table.add_row(str(i), f"{r['score']:.4f}", text)
    console.print(table)

    # show new results surfaced
    vanilla_ids = {r["id"] for r in comparison["vanilla"][:limit]}
    rf_ids = {r["id"] for r in comparison["relevance_feedback"][:limit]}
    new_ids = rf_ids - vanilla_ids
    if new_ids:
        console.print(f"\n[green]{len(new_ids)} new results surfaced by relevance feedback[/green]")
    else:
        console.print(f"\n[dim]Same result set (ordering may differ)[/dim]")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_query_texts(path: str) -> list[str]:
    """Load query texts from a file (plain text, one per line, or JSONL with 'query' field)."""
    import json

    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                queries.append(obj.get("query", line))
            except json.JSONDecodeError:
                queries.append(line)
    return queries


if __name__ == "__main__":
    cli()
