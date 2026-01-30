"""Synthetic query generation via LLM (litellm).

Supported providers (set the corresponding env var):
  - OpenAI:      OPENAI_API_KEY        model="gpt-4o-mini"
  - Anthropic:   ANTHROPIC_API_KEY     model="anthropic/claude-sonnet-4-20250514"
  - OpenRouter:  OPENROUTER_API_KEY    model="openrouter/google/gemma-2-9b-it"
  - Ollama:      (local, no key)       model="ollama/llama3"

See https://docs.litellm.ai/docs/providers for full list.
"""

import json
import logging
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

QUERY_GEN_PROMPT = """You are generating realistic e-commerce search queries that a customer would type to find a product.

Product:
{product_text}

Generate {n} short search queries (1-5 words each) that someone would type into a search engine to find this product. Be realistic: use abbreviations, misspellings, brand names, category terms, and natural shopping language. Do NOT paraphrase the title.

Return ONLY a JSON array of strings, nothing else. Example: ["nike running shoes", "mens athletic shoe", "nike free run"]"""


def generate_synthetic_queries(
    products: list[dict],
    model: str = "gpt-4o-mini",
    queries_per_product: int = 3,
    batch_size: int = 10,
    max_products: Optional[int] = None,
) -> list[dict]:
    """Generate synthetic search queries for products using an LLM.

    Args:
        products: List of product dicts with "text" and "product_id" keys.
        model: litellm model string.
        queries_per_product: Number of queries per product.
        batch_size: Products per LLM call.
        max_products: Limit number of products (None = all).

    Returns:
        List of dicts with "query", "product_id", "positive_text" keys.
    """
    import litellm

    if max_products:
        products = products[:max_products]

    all_pairs = []

    for i in tqdm(range(0, len(products), batch_size), desc="Generating queries"):
        batch = products[i : i + batch_size]

        for product in batch:
            prompt = QUERY_GEN_PROMPT.format(
                product_text=product["text"][:500],
                n=queries_per_product,
            )

            try:
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=200,
                )
                content = response.choices[0].message.content.strip()

                # Parse JSON array
                queries = json.loads(content)
                if not isinstance(queries, list):
                    queries = [str(queries)]

                for q in queries[:queries_per_product]:
                    all_pairs.append({
                        "query": str(q).strip(),
                        "product_id": product["product_id"],
                        "positive_text": product["text"],
                        "positive_ids": {product["product_id"]},
                    })

            except Exception as e:
                logger.warning(f"Failed to generate queries for product {product['product_id']}: {e}")
                continue

    logger.info(f"Generated {len(all_pairs)} synthetic query-product pairs")
    return all_pairs
