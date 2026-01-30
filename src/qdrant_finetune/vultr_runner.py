"""Vultr GPU runner — provisions a GPU instance, uploads data, runs training, and downloads results."""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

VULTR_API_BASE = "https://api.vultr.com/v2"


def _get_api_key() -> str:
    """Get Vultr API key from env."""
    key = os.environ.get("VULTR_API_KEY")
    if not key:
        from dotenv import dotenv_values
        env = dotenv_values(".env")
        key = env.get("VULTR_API_KEY")
    if not key:
        console.print("[red]VULTR_API_KEY not set. Run: qdrant-finetune setup[/red]")
        sys.exit(1)
    return key


def _vultr_request(method: str, path: str, api_key: str, **kwargs) -> requests.Response:
    """Make an authenticated Vultr API request."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{VULTR_API_BASE}{path}"
    resp = requests.request(method, url, headers=headers, **kwargs)
    if resp.status_code >= 400:
        raise RuntimeError(f"Vultr API error {resp.status_code}: {resp.text}")
    return resp


def _find_gpu_plan(api_key: str) -> str:
    """Find an available Vultr GPU plan (prefer cheapest with NVIDIA GPU)."""
    resp = _vultr_request("GET", "/plans", api_key)
    plans = resp.json().get("plans", [])
    gpu_plans = [p for p in plans if p.get("type") == "vcg" and p.get("gpu_vram_gb", 0) > 0]
    if not gpu_plans:
        raise RuntimeError("No GPU plans available on Vultr. Check your account limits.")
    # Sort by price, pick cheapest
    gpu_plans.sort(key=lambda p: p.get("monthly_cost", 9999))
    plan = gpu_plans[0]
    console.print(f"  Using plan: {plan['id']} ({plan.get('gpu_type', 'GPU')}, {plan.get('gpu_vram_gb', '?')}GB VRAM, ${plan.get('monthly_cost', '?')}/mo)")
    return plan["id"]


def _create_instance(api_key: str, plan_id: str, label: str = "qdrant-finetune") -> str:
    """Create a Vultr GPU instance and wait for it to be ready."""
    # Use Ubuntu 22.04 with NVIDIA drivers
    resp = _vultr_request("POST", "/instances", api_key, json={
        "plan": plan_id,
        "region": "ewr",  # New Jersey — adjust as needed
        "os_id": 1743,    # Ubuntu 22.04 LTS
        "label": label,
        "tag": "qdrant-finetune",
        "script_id": None,
        "enable_ipv6": False,
    })
    instance = resp.json()["instance"]
    instance_id = instance["id"]
    console.print(f"  Instance created: {instance_id}")

    # Wait for instance to be ready
    console.print("  Waiting for instance to boot...", end="")
    for _ in range(60):
        time.sleep(10)
        resp = _vultr_request("GET", f"/instances/{instance_id}", api_key)
        status = resp.json()["instance"]["status"]
        power = resp.json()["instance"]["power_status"]
        if status == "active" and power == "running":
            console.print(" [green]ready[/green]")
            ip = resp.json()["instance"]["main_ip"]
            console.print(f"  IP: {ip}")
            return instance_id
        console.print(".", end="")

    raise RuntimeError("Instance failed to start within timeout")


def _get_instance_ip(api_key: str, instance_id: str) -> str:
    """Get the main IP of a Vultr instance."""
    resp = _vultr_request("GET", f"/instances/{instance_id}", api_key)
    return resp.json()["instance"]["main_ip"]


def _ssh_cmd(ip: str, cmd: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a command on the remote instance via SSH."""
    return subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30", f"root@{ip}", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _upload_via_scp(ip: str, local_path: str, remote_path: str):
    """Upload a file to the instance via SCP."""
    subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no", local_path, f"root@{ip}:{remote_path}"],
        check=True,
    )


def _download_via_scp(ip: str, remote_path: str, local_path: str):
    """Download a file from the instance via SCP."""
    subprocess.run(
        ["scp", "-r", "-o", "StrictHostKeyChecking=no", f"root@{ip}:{remote_path}", local_path],
        check=True,
    )


def _destroy_instance(api_key: str, instance_id: str):
    """Destroy a Vultr instance."""
    _vultr_request("DELETE", f"/instances/{instance_id}", api_key)
    console.print(f"  [green]✓[/green] Instance {instance_id} destroyed")


def run_on_vultr(
    data_path: str,
    queries_path: str | None = None,
    config_override: dict | None = None,
):
    """Provision a Vultr GPU instance, upload data, train, download results, destroy instance.

    Args:
        data_path: Local path to product data file.
        queries_path: Local path to query data file (optional).
        config_override: Config overrides dict.
    """
    api_key = _get_api_key()
    instance_id = None

    try:
        # Find a GPU plan
        console.print("\n[yellow]Finding available GPU plan...[/yellow]")
        plan_id = _find_gpu_plan(api_key)

        # Create instance
        console.print("\n[yellow]Provisioning Vultr GPU instance...[/yellow]")
        instance_id = _create_instance(api_key, plan_id)
        ip = _get_instance_ip(api_key, instance_id)

        # Wait for SSH to be ready
        console.print("  Waiting for SSH...", end="")
        for _ in range(30):
            time.sleep(5)
            result = _ssh_cmd(ip, "echo ready", timeout=10)
            if result.returncode == 0:
                console.print(" [green]connected[/green]")
                break
            console.print(".", end="")
        else:
            raise RuntimeError("SSH connection timed out")

        # Install dependencies on the instance
        console.print("\n[yellow]Installing dependencies...[/yellow]")
        _ssh_cmd(ip, "apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1", timeout=300)
        _ssh_cmd(ip, "pip3 install uv > /dev/null 2>&1", timeout=120)
        _ssh_cmd(ip, "uv pip install --system qdrant-sparse-finetune > /dev/null 2>&1", timeout=300)
        console.print("  [green]✓[/green] Dependencies installed")

        # Upload data
        console.print("\n[yellow]Uploading data...[/yellow]")
        _ssh_cmd(ip, "mkdir -p /data /output")
        _upload_via_scp(ip, data_path, f"/data/{Path(data_path).name}")
        console.print(f"  [green]✓[/green] Uploaded {data_path}")

        if queries_path:
            _upload_via_scp(ip, queries_path, f"/data/{Path(queries_path).name}")
            console.print(f"  [green]✓[/green] Uploaded {queries_path}")

        # Set env vars for Qdrant connection
        env_vars = []
        for var in ["QDRANT_URL", "QDRANT_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]:
            val = os.environ.get(var)
            if val:
                env_vars.append(f"export {var}='{val}'")
        env_setup = " && ".join(env_vars) if env_vars else "true"

        # Build training command
        train_cmd = f"qdrant-finetune train --data /data/{Path(data_path).name} --gpu vultr-local --output-dir /output"
        if queries_path:
            train_cmd += f" --queries /data/{Path(queries_path).name}"
        if config_override:
            if "ance_iterations" in config_override:
                train_cmd += f" --ance-iterations {config_override['ance_iterations']}"
            if "batch_size" in config_override:
                train_cmd += f" --batch-size {config_override['batch_size']}"
            if "synth_model" in config_override:
                train_cmd += f" --synth-model {config_override['synth_model']}"

        # Run training
        console.print(f"\n[yellow]Running training on Vultr GPU...[/yellow]")
        console.print(f"  Command: {train_cmd}")
        result = _ssh_cmd(ip, f"{env_setup} && {train_cmd}", timeout=7200)

        if result.returncode != 0:
            console.print(f"[red]Training failed:[/red]\n{result.stderr}")
            raise RuntimeError("Training failed on Vultr")

        console.print(result.stdout)
        console.print("  [green]✓[/green] Training complete")

        # Download results
        console.print("\n[yellow]Downloading trained model...[/yellow]")
        output_dir = config_override.get("output_dir", "./output") if config_override else "./output"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        _download_via_scp(ip, "/output/", output_dir)
        console.print(f"  [green]✓[/green] Model downloaded to {output_dir}")

        console.print(f"\n[bold green]Training complete! Model saved to: {output_dir}[/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise
    finally:
        # Always destroy the instance to avoid charges
        if instance_id:
            console.print("\n[yellow]Cleaning up Vultr instance...[/yellow]")
            try:
                _destroy_instance(api_key, instance_id)
            except Exception as e:
                console.print(f"  [red]Warning: Failed to destroy instance {instance_id}: {e}[/red]")
                console.print(f"  [red]Manually destroy it at https://my.vultr.com/[/red]")
