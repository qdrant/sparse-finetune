"""End-to-end test with a tiny synthetic product catalog."""

import csv
import json
import tempfile
from pathlib import Path

from qdrant_finetune import Trainer, FinetuneConfig


def create_tiny_dataset(tmp_dir: Path):
    """Create a tiny product CSV and queries CSV."""
    products_path = tmp_dir / "products.csv"
    queries_path = tmp_dir / "queries.csv"

    products = [
        {"product_id": "1", "title": "Nike Air Max 90 Running Shoes", "description": "Lightweight running shoes with air cushion sole"},
        {"product_id": "2", "title": "Adidas Ultraboost 22 Sneakers", "description": "Responsive boost midsole for energy return"},
        {"product_id": "3", "title": "Sony WH-1000XM5 Headphones", "description": "Noise cancelling wireless over-ear headphones"},
        {"product_id": "4", "title": "Apple AirPods Pro 2nd Gen", "description": "Active noise cancellation with spatial audio"},
        {"product_id": "5", "title": "Samsung Galaxy S24 Ultra Phone", "description": "6.8 inch display with S Pen and AI features"},
        {"product_id": "6", "title": "Bose QuietComfort Earbuds", "description": "True wireless noise cancelling earbuds"},
        {"product_id": "7", "title": "New Balance 990v6 Running Shoes", "description": "Made in USA premium running shoe"},
        {"product_id": "8", "title": "Google Pixel 8 Pro Smartphone", "description": "AI-powered camera with tensor chip"},
        {"product_id": "9", "title": "JBL Flip 6 Bluetooth Speaker", "description": "Portable waterproof speaker with deep bass"},
        {"product_id": "10", "title": "Under Armour HOVR Phantom 3", "description": "Connected running shoe with energy return"},
    ]

    with open(products_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["product_id", "title", "description"])
        writer.writeheader()
        writer.writerows(products)

    # Query-product relevance pairs (ESCI-style labels)
    query_pairs = [
        {"query_id": "q1", "query": "running shoes", "product_id": "1", "label": "E"},
        {"query_id": "q1", "query": "running shoes", "product_id": "2", "label": "S"},
        {"query_id": "q1", "query": "running shoes", "product_id": "7", "label": "E"},
        {"query_id": "q1", "query": "running shoes", "product_id": "10", "label": "S"},
        {"query_id": "q1", "query": "running shoes", "product_id": "3", "label": "I"},
        {"query_id": "q2", "query": "noise cancelling headphones", "product_id": "3", "label": "E"},
        {"query_id": "q2", "query": "noise cancelling headphones", "product_id": "4", "label": "S"},
        {"query_id": "q2", "query": "noise cancelling headphones", "product_id": "6", "label": "S"},
        {"query_id": "q2", "query": "noise cancelling headphones", "product_id": "9", "label": "C"},
        {"query_id": "q2", "query": "noise cancelling headphones", "product_id": "5", "label": "I"},
        {"query_id": "q3", "query": "smartphone", "product_id": "5", "label": "E"},
        {"query_id": "q3", "query": "smartphone", "product_id": "8", "label": "E"},
        {"query_id": "q3", "query": "smartphone", "product_id": "4", "label": "C"},
        {"query_id": "q3", "query": "smartphone", "product_id": "1", "label": "I"},
    ]

    with open(queries_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query", "product_id", "label"])
        writer.writeheader()
        writer.writerows(query_pairs)

    return str(products_path), str(queries_path)


def main():
    tmp_dir = Path(tempfile.mkdtemp())
    products_path, queries_path = create_tiny_dataset(tmp_dir)

    print(f"Products: {products_path}")
    print(f"Queries: {queries_path}")

    config = FinetuneConfig(
        qdrant_url="http://localhost:6333",
        base_model="distilbert/distilbert-base-uncased",
        architecture="inference_free_splade",
        batch_size=4,
        learning_rate=2e-5,
        num_epochs=1,
        ance_iterations=2,
        mining_top_k=5,
        num_negatives=2,
        epochs_per_ance_iteration=1,
        output_dir=str(tmp_dir / "output"),
        run_name="tiny_test",
        collection_name="tiny_test",
        save_steps=9999,
        logging_steps=1,
    )

    trainer = Trainer(config=config)
    model_path = trainer.fit(data=products_path, queries=queries_path)
    print(f"\nModel saved to: {model_path}")

    # Index and evaluate
    trainer.index(data=products_path, collection_name="tiny_test")
    metrics = trainer.evaluate(queries=queries_path, collection_name="tiny_test")
    print(f"\nFinal metrics: {metrics}")


if __name__ == "__main__":
    main()
