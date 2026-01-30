"""Product text construction for encoding."""

from typing import Dict, List, Optional


def build_product_text(
    title: str,
    brand: str = "",
    description: str = "",
    bullets: Optional[List[str]] = None,
    attributes: Optional[Dict[str, str]] = None,
    max_length: int = 512,
) -> str:
    """Construct product text for encoding. Format: [brand] title | description | bullets | attrs."""
    parts = []
    if brand and brand.strip():
        parts.append(f"[{brand.strip()}]")
    if title and title.strip():
        parts.append(title.strip())
    if description and description.strip():
        parts.append(description.strip()[:200])
    if bullets:
        parts.extend(b.strip() for b in bullets[:3] if b and b.strip())
    if attributes:
        parts.extend(f"{k}={v}" for k, v in list(attributes.items())[:5] if v)

    text = " | ".join(parts)
    if len(text) > max_length * 4:
        text = text[: max_length * 4]
    return text


def build_query_text(query: str) -> str:
    return query.strip()
