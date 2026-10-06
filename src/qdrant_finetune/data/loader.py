"""Generic data loader for CSV, JSON, JSONL, and HuggingFace datasets."""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Common product text column names, ordered by priority
TEXT_COLUMN_CANDIDATES = [
    "title", "product_title", "name", "product_name",
    "description", "product_description",
    "text", "content",
]

QUERY_COLUMN_CANDIDATES = ["query", "search_query", "query_text", "question"]
ID_COLUMN_CANDIDATES = ["product_id", "id", "asin", "sku", "item_id"]
LABEL_COLUMN_CANDIDATES = ["esci_label", "label", "relevance", "relevance_label", "grade"]


def _detect_columns(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find the first matching column name."""
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def load_products(
    path: str,
    text_fields: Optional[list[str]] = None,
    id_field: Optional[str] = None,
) -> list[dict]:
    """Load product data from CSV, JSON, JSONL, or HuggingFace dataset name.

    Returns list of dicts with keys: product_id, text, and original fields as payload.
    """
    path_str = str(path)

    if path_str.endswith(".csv"):
        df = pd.read_csv(path_str)
    elif path_str.endswith(".json"):
        df = pd.read_json(path_str)
    elif path_str.endswith(".jsonl"):
        df = pd.read_json(path_str, lines=True)
    else:
        # Try HuggingFace datasets
        from datasets import load_dataset
        ds = load_dataset(path_str, split="train")
        df = ds.to_pandas()

    logger.info(f"Loaded {len(df)} products from {path_str}")

    # Auto-detect ID column
    id_col = id_field or _detect_columns(df, ID_COLUMN_CANDIDATES)
    if id_col is None:
        df["_auto_id"] = [str(i) for i in range(len(df))]
        id_col = "_auto_id"

    # Auto-detect text columns
    if text_fields is None:
        text_fields = []
        for candidate in TEXT_COLUMN_CANDIDATES:
            col = _detect_columns(df, [candidate])
            if col and col not in text_fields:
                text_fields.append(col)
        if not text_fields:
            raise ValueError(
                f"Could not auto-detect text columns. Columns: {list(df.columns)}. "
                f"Pass text_fields explicitly."
            )

    logger.info(f"Using text fields: {text_fields}, ID field: {id_col}")

    products = []
    for _, row in df.iterrows():
        text_parts = [str(row[f]) for f in text_fields if pd.notna(row.get(f))]
        products.append({
            "product_id": str(row[id_col]),
            "text": " | ".join(text_parts),
            "payload": {k: str(v) for k, v in row.to_dict().items() if pd.notna(v)},
        })

    return products


def load_queries(
    path: str,
    query_field: Optional[str] = None,
    id_field: Optional[str] = None,
    product_id_field: Optional[str] = None,
    label_field: Optional[str] = None,
) -> list[dict]:
    """Load query data with relevance judgments.

    Returns list of dicts with keys: query_id, query, relevance (dict of product_id -> label).
    """
    path_str = str(path)

    if path_str.endswith(".csv"):
        df = pd.read_csv(path_str)
    elif path_str.endswith(".json"):
        df = pd.read_json(path_str)
    elif path_str.endswith(".jsonl"):
        df = pd.read_json(path_str, lines=True)
    else:
        from datasets import load_dataset
        ds = load_dataset(path_str, split="test")
        df = ds.to_pandas()

    logger.info(f"Loaded {len(df)} query-product pairs from {path_str}")

    query_col = query_field or _detect_columns(df, QUERY_COLUMN_CANDIDATES)
    id_col = id_field or _detect_columns(df, ["query_id"] + ID_COLUMN_CANDIDATES)
    pid_col = product_id_field or _detect_columns(df, ["product_id", "asin", "doc_id", "item_id"])
    label_col = label_field or _detect_columns(df, LABEL_COLUMN_CANDIDATES)

    if not query_col:
        raise ValueError(f"Could not detect query column. Columns: {list(df.columns)}")
    if not pid_col:
        raise ValueError(f"Could not detect product ID column. Columns: {list(df.columns)}")

    # Check for synthetic query format (has positive_text and positive_ids)
    if "positive_text" in df.columns and "positive_ids" in df.columns:
        logger.info("Detected synthetic query format with positive_text/positive_ids")
        results = []
        for i, row in df.iterrows():
            pids = row["positive_ids"]
            if isinstance(pids, str):
                import json as _json
                try:
                    pids = _json.loads(pids)
                except Exception:
                    pids = [pids]
            results.append({
                "query_id": str(i),
                "query": str(row[query_col]),
                "positive_text": str(row["positive_text"]),
                "positive_ids": [str(p) for p in pids],
                "relevance": {str(p): "E" for p in pids},
            })
        return results

    # Group by query (standard relevance-judgment format)
    queries_map: dict = {}
    for _, row in df.iterrows():
        qtext = str(row[query_col])
        qid = str(row[id_col]) if id_col else qtext
        pid = str(row[pid_col])
        label = row[label_col] if label_col and pd.notna(row.get(label_col)) else None

        if qid not in queries_map:
            queries_map[qid] = {"query_id": qid, "query": qtext, "relevance": {}}
        if label is not None:
            queries_map[qid]["relevance"][pid] = label

    return list(queries_map.values())
