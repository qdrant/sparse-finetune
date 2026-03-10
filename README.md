# qdrant-sparse-finetune

Fine-tune SPLADE sparse embeddings for any product catalog with ANCE hard negative mining via Qdrant.

## One line

```python
from qdrant_finetune import finetune

model_path = finetune("products.csv")
```

That's it. Loads your data, generates synthetic queries via LLM, trains a SPLADE encoder with 3 rounds of ANCE hard negative mining against Qdrant, and saves the model.

## Setup

```bash
pip install git+https://github.com/qdrant/sparse-finetune.git
qdrant-finetune setup
```

The setup wizard walks you through:
1. **Qdrant** — URL + API key (validates the connection)
2. **LLM provider** — for synthetic query generation (OpenAI, Anthropic, OpenRouter, Ollama)
3. **Modal** — optional cloud GPU setup (creates secrets, volumes automatically)

Writes a `.env` file so you never have to pass creds again.

## Train

```bash
# Local GPU (or CPU — it just works, GPU is faster)
qdrant-finetune train --data products.csv

# With your own labeled queries
qdrant-finetune train --data products.csv --queries queries.csv

# On Modal cloud GPU
qdrant-finetune train --data products.csv --gpu modal
```

`--gpu modal` handles everything: uploads your data, creates Modal volumes, launches an A10G, runs training, and tells you where to download the model.

## Python API

```python
from qdrant_finetune import finetune

# Simplest — uses .env for creds, generates synthetic queries
model_path = finetune("products.csv")

# With labeled queries
model_path = finetune("products.csv", queries="queries.csv")

# Full control
model_path = finetune(
    "products.csv",
    queries="queries.csv",
    qdrant_url="https://my-cluster.qdrant.io",
    qdrant_api_key="...",
    base_model="distilbert/distilbert-base-uncased",
    ance_iterations=5,
    batch_size=64,
    synth_model="ollama/llama3",
)
```

Or use the `Trainer` class for step-by-step control:

```python
from qdrant_finetune import Trainer, FinetuneConfig

trainer = Trainer(FinetuneConfig(ance_iterations=3))
trainer.fit(data="products.csv")
trainer.index(data="products.csv", collection_name="my_products")
trainer.evaluate(queries="test_queries.csv")
trainer.export("./my_model")
```

## How It Works

```
Product Data ──► Create SPLADE Encoder
                        │
  Queries ──────────────┤  (labeled or synthetic via LLM)
                        ▼
                 ANCE Training Loop ◄──┐
                  1. Train model       │
                  2. Index in Qdrant   │
                  3. Mine negatives    │  × N iterations
                  4. Retrain ──────────┘
                        │
                        ▼
               Fine-tuned Model + Qdrant Collection
```

## LLM Providers

Synthetic query generation uses [litellm](https://docs.litellm.ai/docs/providers). Set the env var, pass the model string:

| Provider | Env Var | Example |
|----------|---------|---------|
| OpenAI | `OPENAI_API_KEY` | `--synth-model gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `--synth-model anthropic/claude-sonnet-4-20250514` |
| OpenRouter | `OPENROUTER_API_KEY` | `--synth-model openrouter/google/gemma-2-9b-it` |
| Ollama (local) | — | `--synth-model ollama/llama3` |

## CLI Reference

| Command | Description |
|---------|-------------|
| `qdrant-finetune setup` | Interactive setup wizard |
| `qdrant-finetune train` | Full training pipeline with ANCE |
| `qdrant-finetune evaluate` | Evaluate model (nDCG, MRR, Recall, Precision) |
| `qdrant-finetune index` | Index products into Qdrant |
| `qdrant-finetune generate-queries` | Generate synthetic queries via LLM |

## Configuration

Set via CLI flags, `.env` file, environment variables (`QDRANT_FINETUNE_` prefix), or YAML:

```yaml
# config.yaml
qdrant_url: "localhost:6333"
base_model: "distilbert/distilbert-base-uncased"
architecture: "inference_free_splade"
batch_size: 32
learning_rate: 2e-5
ance_iterations: 3
mining_top_k: 20
num_negatives: 3
synth_model: "gpt-4o-mini"
```

## Data Formats

**Products** (CSV, JSON, JSONL, or HuggingFace dataset):
- Auto-detects columns: `title`, `description`, `name`, `product_id`, etc.
- Or specify: `--text-fields title,description`

**Queries** (CSV, JSON, JSONL):
- Columns: `query`, `product_id`, `label` (e.g., E/S/C/I for ESCI, or numeric grades)

## Requirements

- Python 3.10+
- A running Qdrant instance (local Docker or Qdrant Cloud)
- GPU recommended but not required (CPU training works, just slower)
- For synthetic queries: an LLM API key or local Ollama
