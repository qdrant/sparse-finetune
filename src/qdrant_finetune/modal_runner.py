"""Modal GPU runner — handles secrets, volumes, data upload, and remote execution."""

import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def _run_modal_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a modal CLI command."""
    result = subprocess.run(
        ["modal", *args],
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"modal {' '.join(args)} failed:\n{result.stderr}")
    return result


def _check_modal_installed():
    """Check if modal CLI is installed and authenticated."""
    result = subprocess.run(["modal", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        console.print("[red]Modal CLI not found. Install it:[/red]")
        console.print("  pip install modal")
        console.print("  modal setup")
        sys.exit(1)


def _ensure_secret(name: str, env_vars: dict[str, str]):
    """Create or update a Modal secret."""
    # Check if secret exists
    result = _run_modal_cmd(["secret", "list"], check=False)
    if name in result.stdout:
        console.print(f"  Secret '{name}' already exists, updating...")
        cmd = ["secret", "create", name, "--force"]
    else:
        cmd = ["secret", "create", name]

    for key, value in env_vars.items():
        cmd.append(f"{key}={value}")

    _run_modal_cmd(cmd)
    console.print(f"  [green]✓[/green] Secret '{name}' configured")


def _ensure_volume(name: str):
    """Create a Modal volume if it doesn't exist."""
    result = _run_modal_cmd(["volume", "list"], check=False)
    if name not in result.stdout:
        _run_modal_cmd(["volume", "create", name])
    console.print(f"  [green]✓[/green] Volume '{name}' ready")


def _upload_file(volume_name: str, local_path: str, remote_path: str):
    """Upload a local file to a Modal volume."""
    _run_modal_cmd(["volume", "put", volume_name, local_path, remote_path, "--force"])
    console.print(f"  [green]✓[/green] Uploaded {local_path} → {remote_path}")


def setup_modal(qdrant_url: str, qdrant_api_key: str, llm_env: dict[str, str] | None = None):
    """Set up Modal secrets and volumes for training.

    Args:
        qdrant_url: Qdrant cluster URL.
        qdrant_api_key: Qdrant API key.
        llm_env: Dict of LLM env vars (e.g. {"OPENAI_API_KEY": "sk-..."}).
    """
    _check_modal_installed()

    console.print("\n[bold cyan]Setting up Modal...[/bold cyan]")

    # Qdrant secret
    console.print("\n[yellow]Configuring Qdrant secret...[/yellow]")
    _ensure_secret("qdrant-secret", {
        "QDRANT_URL": qdrant_url,
        "QDRANT_API_KEY": qdrant_api_key,
    })

    # LLM secret (if provided)
    if llm_env:
        console.print("[yellow]Configuring LLM secret...[/yellow]")
        _ensure_secret("llm-secret", llm_env)

    # Volumes
    console.print("\n[yellow]Creating volumes...[/yellow]")
    _ensure_volume("finetune-data")
    _ensure_volume("finetune-output")

    console.print("\n[bold green]Modal setup complete![/bold green]")


def run_on_modal(
    data_path: str,
    queries_path: str | None = None,
    config_override: dict | None = None,
    gpu: str = "A10G",
):
    """Upload data and run training on Modal.

    Args:
        data_path: Local path to product data file.
        queries_path: Local path to query data file (optional).
        config_override: Config overrides dict.
        gpu: Modal GPU type (T4, A10G, A100, H100).
    """
    _check_modal_installed()

    # Ensure volumes exist
    _ensure_volume("finetune-data")
    _ensure_volume("finetune-output")

    # Upload data files
    console.print("\n[yellow]Uploading data to Modal...[/yellow]")
    data_filename = Path(data_path).name
    remote_data = f"/{data_filename}"
    _upload_file("finetune-data", data_path, remote_data)

    remote_queries = None
    if queries_path:
        queries_filename = Path(queries_path).name
        remote_queries = f"/{queries_filename}"
        _upload_file("finetune-data", queries_path, remote_queries)

    # Build the modal run command
    console.print(f"\n[yellow]Launching training on Modal ({gpu} GPU)...[/yellow]")

    # Write a temp config for the modal app to pick up
    modal_app_path = _find_modal_app()

    cmd = [
        "modal", "run", str(modal_app_path),
        "--data", f"/data{remote_data}",
    ]
    if remote_queries:
        cmd.extend(["--queries", f"/data{remote_queries}"])

    if config_override:
        if "ance_iterations" in config_override:
            cmd.extend(["--ance-iterations", str(config_override["ance_iterations"])])
        if "batch_size" in config_override:
            cmd.extend(["--batch-size", str(config_override["batch_size"])])
        if "synth_model" in config_override:
            cmd.extend(["--synth-model", config_override["synth_model"]])

    console.print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        console.print("\n[bold green]Training complete![/bold green]")
        # Download into output/<run_name>/final, where evaluate and publish look for the model
        dest = shlex.quote(f"output/{(config_override or {}).get('run_name', 'finetune')}")
        console.print(f"Download model: mkdir -p {dest} && modal volume get finetune-output /modal_run/final {dest}")
    else:
        console.print("\n[bold red]Training failed. Check Modal logs.[/bold red]")
        sys.exit(1)


def _find_modal_app() -> Path:
    """Find the modal_app.py file."""
    # Check current directory
    local = Path("modal_app.py")
    if local.exists():
        return local

    # Check package directory
    pkg_dir = Path(__file__).parent.parent.parent
    pkg_app = pkg_dir / "modal_app.py"
    if pkg_app.exists():
        return pkg_app

    # Check installed package location
    import importlib.resources
    try:
        ref = importlib.resources.files("qdrant_finetune").joinpath("../../modal_app.py")
        if Path(str(ref)).exists():
            return Path(str(ref))
    except Exception:
        pass

    raise FileNotFoundError(
        "modal_app.py not found. Make sure you're running from the qdrant-sparse-finetune directory, "
        "or copy modal_app.py to your working directory."
    )
