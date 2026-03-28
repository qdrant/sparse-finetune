"""Integration test for relevance feedback using Qdrant in-memory mode.

No external Qdrant server or Docker needed.

Note: Qdrant's local (in-memory) client doesn't support RelevanceFeedbackQuery
with sparse vectors due to a type handling limitation. These tests use dense
vectors to validate the RF logic end-to-end. Sparse vector + RF works correctly
against a real Qdrant server (v1.17+).
"""

import numpy as np
from qdrant_client import QdrantClient, models

from qdrant_finetune.relevance_feedback import (
    RFConfig,
    RFParams,
    RelevanceFeedbackSearch,
)


class FakeSparseEncoder:
    """Encoder mock that produces dense vectors (used as 'sparse' stand-in for local client)."""

    def __init__(self, dim=64):
        self.dim = dim

    def encode(self, texts, **kwargs):
        results = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            results.append(vec)
        return results


class FakeDenseEncoder:
    """Fake dense embedding model (replaces fastembed TextEmbedding for tests)."""

    def __init__(self, dim=64):
        self.dim = dim

    def embed(self, texts):
        for text in texts:
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            yield vec


PRODUCTS = [
    {"product_id": "1", "text": "Nike Air Max 90 Running Shoes black"},
    {"product_id": "2", "text": "Adidas Ultraboost 22 Sneakers white"},
    {"product_id": "3", "text": "Sony WH-1000XM5 Noise Cancelling Headphones"},
    {"product_id": "4", "text": "Apple AirPods Pro 2nd Gen wireless earbuds"},
    {"product_id": "5", "text": "Samsung Galaxy S24 Ultra Smartphone"},
    {"product_id": "6", "text": "Bose QuietComfort Earbuds noise cancelling"},
    {"product_id": "7", "text": "New Balance 990v6 Running Shoes grey"},
    {"product_id": "8", "text": "Google Pixel 8 Pro Smartphone camera"},
    {"product_id": "9", "text": "JBL Flip 6 Bluetooth Speaker portable"},
    {"product_id": "10", "text": "Under Armour HOVR Phantom Running Shoes"},
]

DIM = 64
VECTOR_NAME = "text"


def _setup_collection():
    """Create an in-memory Qdrant collection with dense vectors and index products."""
    client = QdrantClient(":memory:")
    collection_name = "test_rf"
    encoder = FakeSparseEncoder(dim=DIM)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            VECTOR_NAME: models.VectorParams(size=DIM, distance=models.Distance.COSINE),
        },
    )

    texts = [p["text"] for p in PRODUCTS]
    embeddings = encoder.encode(texts)

    points = []
    for i, (product, emb) in enumerate(zip(PRODUCTS, embeddings)):
        points.append(models.PointStruct(
            id=i,
            vector={VECTOR_NAME: emb.tolist()},
            payload={"product_id": product["product_id"], "text": product["text"]},
        ))

    client.upsert(collection_name=collection_name, points=points)
    feedback = FakeDenseEncoder(dim=DIM)
    return client, collection_name, encoder, feedback


def _make_rf_search(client, collection_name, encoder, feedback, **config_kwargs):
    """Helper to create RelevanceFeedbackSearch with fake feedback model injected."""
    config = RFConfig(sparse_vector_name=VECTOR_NAME, **config_kwargs)
    rf = RelevanceFeedbackSearch(
        client=client,
        splade_model=encoder,
        collection_name=collection_name,
        config=config,
    )
    rf.feedback_model = feedback
    return rf


def test_initial_retrieval():
    """Basic retrieval should return results."""
    client, collection_name, encoder, feedback = _setup_collection()
    rf_search = _make_rf_search(client, collection_name, encoder, feedback)

    results = rf_search._initial_retrieval("running shoes", limit=5)
    assert len(results) > 0
    assert len(results) <= 5
    for r in results:
        assert "product_id" in r.payload


def test_feedback_scoring():
    """Feedback model should produce scores for all results."""
    client, collection_name, encoder, feedback = _setup_collection()
    rf_search = _make_rf_search(client, collection_name, encoder, feedback)

    results = rf_search._initial_retrieval("headphones", limit=5)
    scores = rf_search._score_with_feedback("headphones", results)

    assert len(scores) == len(results)
    for s in scores:
        assert isinstance(s, float)
        assert -1.1 <= s <= 1.1


def test_search_compare_returns_both():
    """search_compare should return both vanilla and RF results."""
    client, collection_name, encoder, feedback = _setup_collection()
    rf_search = _make_rf_search(client, collection_name, encoder, feedback, context_limit=3, rf_limit=5)

    comparison = rf_search.search_compare("smartphone", limit=5, context_limit=3)

    assert "vanilla" in comparison
    assert "relevance_feedback" in comparison
    assert "query" in comparison
    assert comparison["query"] == "smartphone"
    assert len(comparison["vanilla"]) > 0
    assert len(comparison["relevance_feedback"]) > 0


def test_rf_search_end_to_end():
    """Full two-pass search should return results."""
    client, collection_name, encoder, feedback = _setup_collection()
    rf_search = _make_rf_search(
        client, collection_name, encoder, feedback,
        context_limit=3, rf_limit=5,
        params=RFParams(a=0.3, b=1.0, c=0.5),
    )

    results = rf_search.search("running shoes", limit=5, context_limit=3)
    assert len(results) > 0
    assert len(results) <= 5


def test_rf_with_different_params():
    """Different RF params should produce results (may differ)."""
    client, collection_name, encoder, feedback = _setup_collection()

    search_a = _make_rf_search(
        client, collection_name, encoder, feedback,
        context_limit=3, rf_limit=5,
        params=RFParams(a=0.1, b=0.5, c=0.9),
    )
    search_b = _make_rf_search(
        client, collection_name, encoder, feedback,
        context_limit=3, rf_limit=5,
        params=RFParams(a=0.9, b=0.1, c=0.1),
    )

    results_a = search_a.search("earbuds", limit=5, context_limit=3)
    results_b = search_b.search("earbuds", limit=5, context_limit=3)

    assert len(results_a) > 0
    assert len(results_b) > 0


def test_rf_surfaces_new_results():
    """RF should potentially surface different results than vanilla."""
    client, collection_name, encoder, feedback = _setup_collection()
    rf_search = _make_rf_search(
        client, collection_name, encoder, feedback,
        context_limit=3, rf_limit=10,
        params=RFParams(a=0.1, b=1.0, c=2.0),  # heavy feedback weight
    )

    comparison = rf_search.search_compare("running shoes", limit=10, context_limit=3)

    vanilla_ids = [r["id"] for r in comparison["vanilla"]]
    rf_ids = [r["id"] for r in comparison["relevance_feedback"]]

    # both should return results
    assert len(vanilla_ids) > 0
    assert len(rf_ids) > 0
    # ordering or set may differ (not guaranteed, but the code path is exercised)
