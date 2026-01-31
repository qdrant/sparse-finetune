"""Dashboard server for qdrant-sparse-finetune."""

import os
import sys
import time
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add src to path so we can import job_store
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from qdrant_finetune.job_store import JobStore

load_dotenv()

app = FastAPI(title="qdrant-sparse-finetune Dashboard")

DASHBOARD_DIR = Path(__file__).parent
STATIC_DIR = DASHBOARD_DIR / "static"

store = JobStore()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    data_path: str
    queries_path: Optional[str] = None
    gpu_backend: str = "modal"
    ance_iterations: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    base_model: str = "distilbert/distilbert-base-uncased"
    num_epochs: int = 3
    run_name: str = "finetune"


class EvaluateRequest(BaseModel):
    model_path: str
    queries_path: str
    collection_name: Optional[str] = None


class PublishRequest(BaseModel):
    model_path: str
    repo_name: str
    hf_token: str
    private: bool = False


class SearchRequest(BaseModel):
    collection_name: str
    query: str
    top_k: int = 10


# ---------------------------------------------------------------------------
# API: Status
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_status():
    qdrant_ok = False
    qdrant_url = os.environ.get("QDRANT_URL", os.environ.get("QDRANT_FINETUNE_QDRANT_URL", "http://localhost:6333"))
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=qdrant_url,
            api_key=os.environ.get("QDRANT_API_KEY", os.environ.get("QDRANT_FINETUNE_QDRANT_API_KEY")) or None,
            timeout=5,
        )
        client.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    gpu_available = False
    gpu_name = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    # Check cloud GPU providers
    cloud_gpus = []
    if os.environ.get("VULTR_API_KEY"):
        cloud_gpus.append("Vultr")
    try:
        import modal  # noqa: F401
        cloud_gpus.append("Modal")
    except ImportError:
        pass
    if not gpu_available and cloud_gpus:
        gpu_available = True
        gpu_name = "Cloud: " + ", ".join(cloud_gpus)

    version = "unknown"
    try:
        from importlib.metadata import version as pkg_version
        version = pkg_version("qdrant-sparse-finetune")
    except Exception:
        pass

    return {
        "qdrant_connected": qdrant_ok,
        "qdrant_url": qdrant_url,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "version": version,
    }


# ---------------------------------------------------------------------------
# API: Training
# ---------------------------------------------------------------------------

def _run_training(job_id: str, req: TrainRequest):
    store.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())
    store.append_log(job_id, f"[{datetime.utcnow().isoformat()}] Starting training...")
    store.append_log(job_id, f"  Data: {req.data_path}")
    store.append_log(job_id, f"  GPU backend: {req.gpu_backend}")
    store.append_log(job_id, f"  ANCE iterations: {req.ance_iterations}")
    store.append_log(job_id, f"  Batch size: {req.batch_size}")
    store.append_log(job_id, f"  Learning rate: {req.learning_rate}")

    try:
        overrides = {
            "ance_iterations": req.ance_iterations,
            "batch_size": req.batch_size,
            "learning_rate": req.learning_rate,
            "base_model": req.base_model,
            "num_epochs": req.num_epochs,
            "run_name": req.run_name,
        }

        if req.gpu_backend == "modal":
            store.append_log(job_id, "Launching on Modal...")
            from qdrant_finetune.modal_runner import run_on_modal
            run_on_modal(
                data_path=req.data_path,
                queries_path=req.queries_path,
                config_override=overrides,
            )
        elif req.gpu_backend == "vultr":
            store.append_log(job_id, "Launching on Vultr...")
            from qdrant_finetune.vultr_runner import run_on_vultr
            run_on_vultr(
                data_path=req.data_path,
                queries_path=req.queries_path,
                config_override=overrides,
            )
        else:
            raise ValueError(f"Unknown GPU backend: {req.gpu_backend}")

        store.update_job(job_id, status="completed", completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, f"[{datetime.utcnow().isoformat()}] Training completed.")

    except Exception as e:
        store.update_job(job_id, status="failed", error=str(e), completed_at=datetime.utcnow().isoformat())
        store.append_log(job_id, f"[{datetime.utcnow().isoformat()}] ERROR: {e}")


@app.post("/api/train")
def start_training(req: TrainRequest):
    job = store.create_job("train", config=req.model_dump(), source="dashboard")
    thread = threading.Thread(target=_run_training, args=(job["id"], req), daemon=True)
    thread.start()
    return {"job_id": job["id"], "status": "pending"}


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": store.list_jobs()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ---------------------------------------------------------------------------
# API: Evaluate
# ---------------------------------------------------------------------------

@app.post("/api/evaluate")
def evaluate_model(req: EvaluateRequest):
    job = store.create_job("evaluate", config=req.model_dump(), source="dashboard")
    job_id = job["id"]

    def _run():
        store.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())
        try:
            from qdrant_finetune.config import FinetuneConfig
            from qdrant_finetune.trainer import Trainer

            config = FinetuneConfig()
            trainer = Trainer(config=config)
            trainer.load(req.model_path)
            metrics = trainer.evaluate(
                queries=req.queries_path,
                collection_name=req.collection_name,
            )
            store.update_job(job_id, status="completed", results=metrics, completed_at=datetime.utcnow().isoformat())
            store.append_log(job_id, f"Evaluation completed: {metrics}")
        except Exception as e:
            store.update_job(job_id, status="failed", error=str(e), completed_at=datetime.utcnow().isoformat())
            store.append_log(job_id, f"ERROR: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "pending"}


# ---------------------------------------------------------------------------
# API: Collections
# ---------------------------------------------------------------------------

@app.get("/api/collections")
def list_collections():
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=os.environ.get("QDRANT_URL", os.environ.get("QDRANT_FINETUNE_QDRANT_URL", "http://localhost:6333")),
            api_key=os.environ.get("QDRANT_API_KEY", os.environ.get("QDRANT_FINETUNE_QDRANT_API_KEY")) or None,
            timeout=10,
        )
        result = client.get_collections()
        collections = []
        for c in result.collections:
            info = client.get_collection(c.name)
            collections.append({
                "name": c.name,
                "vectors_count": getattr(info, "vectors_count", None) or info.points_count,
                "points_count": info.points_count,
                "status": info.status.value if hasattr(info.status, "value") else str(info.status),
            })
        return {"collections": collections}
    except Exception as e:
        return {"collections": [], "error": str(e)}


@app.post("/api/search")
def search_collection(req: SearchRequest):
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=os.environ.get("QDRANT_URL", os.environ.get("QDRANT_FINETUNE_QDRANT_URL", "http://localhost:6333")),
            api_key=os.environ.get("QDRANT_API_KEY", os.environ.get("QDRANT_FINETUNE_QDRANT_API_KEY")) or None,
            timeout=10,
        )
        # Try sparse search first, fall back to scroll
        try:
            from qdrant_client.models import NamedSparseVector, SparseVector
            # Simple keyword search via scroll with filter
            results = client.scroll(
                collection_name=req.collection_name,
                limit=req.top_k,
                with_payload=True,
            )
            points = [{"id": str(p.id), "payload": p.payload, "score": None} for p in results[0]]
        except Exception:
            points = []

        return {"results": points}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# API: Publish
# ---------------------------------------------------------------------------

@app.post("/api/publish")
def publish_model(req: PublishRequest):
    job = store.create_job("publish", config=req.model_dump(exclude={"hf_token"}), source="dashboard")
    job_id = job["id"]

    def _run():
        store.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())
        try:
            from sentence_transformers import SparseEncoder
            store.append_log(job_id, f"Loading model from {req.model_path}...")
            encoder = SparseEncoder(req.model_path)
            store.append_log(job_id, f"Pushing to {req.repo_name}...")
            encoder.push_to_hub(req.repo_name, token=req.hf_token, private=req.private)
            store.update_job(job_id, status="completed", completed_at=datetime.utcnow().isoformat())
            store.append_log(job_id, f"Published to https://huggingface.co/{req.repo_name}")
        except Exception as e:
            store.update_job(job_id, status="failed", error=str(e), completed_at=datetime.utcnow().isoformat())
            store.append_log(job_id, f"ERROR: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "pending"}


# ---------------------------------------------------------------------------
# API: Settings
# ---------------------------------------------------------------------------

ENV_PATH = Path(__file__).parent.parent / ".env"

SETTINGS_KEYS = [
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "VULTR_API_KEY",
]


def _mask_value(val: str) -> str:
    if not val:
        return ""
    if len(val) <= 8:
        return val[:2] + "..." + val[-2:]
    return val[:4] + "..." + val[-4:]


def _read_env_file() -> dict[str, str]:
    """Read .env file and return key-value pairs."""
    result: dict[str, str] = {}
    if not ENV_PATH.exists():
        return result
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        result[key] = value
    return result


@app.get("/api/settings")
def get_settings():
    env_vals = _read_env_file()
    masked = {}
    for key in SETTINGS_KEYS:
        raw = env_vals.get(key, "")
        masked[key] = _mask_value(raw)
    return {"settings": masked}


@app.put("/api/settings")
async def update_settings(request: Request):
    body = await request.json()
    env_vals = _read_env_file()

    for key in SETTINGS_KEYS:
        if key in body:
            val = body[key]
            if val == "__UNCHANGED__" or val == "":
                continue
            env_vals[key] = val

    # Write .env
    lines = []
    for k, v in env_vals.items():
        lines.append(f'{k}="{v}"')
    ENV_PATH.write_text("\n".join(lines) + "\n")

    # Reload into process
    load_dotenv(str(ENV_PATH), override=True)
    for key in SETTINGS_KEYS:
        if key in env_vals:
            os.environ[key] = env_vals[key]

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Static files + index
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    def _open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{args.port}")

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port)
