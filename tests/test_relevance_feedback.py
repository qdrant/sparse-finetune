"""Unit tests for relevance feedback module (no Qdrant or GPU required)."""

import pytest

from qdrant_finetune.relevance_feedback import RFConfig, RFParams


class TestRFParams:
    def test_defaults(self):
        params = RFParams()
        assert params.a == 0.25
        assert params.b == 1.0
        assert params.c == 0.5

    def test_to_dict(self):
        params = RFParams(a=0.12, b=0.43, c=0.03)
        d = params.to_dict()
        assert d == {"a": 0.12, "b": 0.43, "c": 0.03}

    def test_from_dict(self):
        d = {"a": 0.24, "b": 1.35, "c": 0.59}
        params = RFParams.from_dict(d)
        assert params.a == 0.24
        assert params.b == 1.35
        assert params.c == 0.59

    def test_roundtrip(self):
        original = RFParams(a=0.1, b=2.0, c=0.8)
        restored = RFParams.from_dict(original.to_dict())
        assert original.a == restored.a
        assert original.b == restored.b
        assert original.c == restored.c


class TestRFConfig:
    def test_defaults(self):
        config = RFConfig()
        assert config.feedback_model == "mixedbread-ai/mxbai-embed-large-v1"
        assert config.context_limit == 10
        assert config.rf_limit == 10
        assert config.sparse_vector_name == "text"
        assert config.dense_vector_name == "feedback"
        assert config.dense_vector_size == 1024
        assert isinstance(config.params, RFParams)

    def test_custom_config(self):
        params = RFParams(a=0.3, b=0.5, c=0.7)
        config = RFConfig(
            feedback_model="colbert-ir/colbertv2.0",
            context_limit=5,
            rf_limit=20,
            params=params,
        )
        assert config.feedback_model == "colbert-ir/colbertv2.0"
        assert config.context_limit == 5
        assert config.rf_limit == 20
        assert config.params.a == 0.3


class TestCLIHelpers:
    def test_load_query_texts_plain(self, tmp_path):
        """Plain text file, one query per line."""
        f = tmp_path / "queries.txt"
        f.write_text("running shoes\nnoise cancelling headphones\nsmartphone\n")

        from qdrant_finetune.cli import _load_query_texts

        queries = _load_query_texts(str(f))
        assert queries == ["running shoes", "noise cancelling headphones", "smartphone"]

    def test_load_query_texts_jsonl(self, tmp_path):
        """JSONL file with 'query' field."""
        import json

        f = tmp_path / "queries.jsonl"
        lines = [
            json.dumps({"query": "blue running shoes", "id": 1}),
            json.dumps({"query": "wireless earbuds", "id": 2}),
        ]
        f.write_text("\n".join(lines) + "\n")

        from qdrant_finetune.cli import _load_query_texts

        queries = _load_query_texts(str(f))
        assert queries == ["blue running shoes", "wireless earbuds"]

    def test_load_query_texts_empty_lines(self, tmp_path):
        """Blank lines should be skipped."""
        f = tmp_path / "queries.txt"
        f.write_text("query one\n\nquery two\n\n\n")

        from qdrant_finetune.cli import _load_query_texts

        queries = _load_query_texts(str(f))
        assert queries == ["query one", "query two"]
