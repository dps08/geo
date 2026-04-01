"""Extract brand mentions, positions, and frequency from LLM responses."""

import re
from collections import defaultdict


def extract_brands_from_response(response_text: str, tracked_brands: dict) -> dict:
    """Extract which tracked brands are mentioned in an LLM response and their positions.

    Args:
        response_text: The LLM's response text.
        tracked_brands: Dict with 'large' and 'small' brand lists.

    Returns:
        Dict with brand mention details.
    """
    if not response_text:
        return {
            "brands_mentioned": [],
            "brand_positions": {},
            "brand_count": 0,
            "mentions_by_size": {"large": [], "small": []},
        }

    all_brands = tracked_brands.get("large", []) + tracked_brands.get("small", [])
    response_lower = response_text.lower()

    # Find all brand mentions and their first character position in the text
    brand_mentions = {}
    for brand in all_brands:
        brand_lower = brand.lower()

        # Try exact match first, then partial match
        patterns = [
            re.escape(brand_lower),
            re.escape(brand_lower.replace(".", "")),  # Handle "Monday.com" -> "monday com"
        ]
        # Also handle common abbreviations
        if " " in brand:
            patterns.append(re.escape(brand_lower.split()[0]))  # First word match

        for pattern in patterns:
            match = re.search(r'\b' + pattern + r'\b', response_lower)
            if match:
                brand_mentions[brand] = match.start()
                break

    # Sort brands by position in text (earlier = higher ranking)
    sorted_brands = sorted(brand_mentions.items(), key=lambda x: x[1])
    brand_positions = {brand: pos + 1 for pos, (brand, _) in enumerate(sorted_brands)}

    # Categorize by size
    large_brands = tracked_brands.get("large", [])
    small_brands = tracked_brands.get("small", [])
    mentions_large = [b for b in brand_mentions if b in large_brands]
    mentions_small = [b for b in brand_mentions if b in small_brands]

    return {
        "brands_mentioned": list(brand_mentions.keys()),
        "brand_positions": brand_positions,
        "brand_count": len(brand_mentions),
        "mentions_by_size": {"large": mentions_large, "small": mentions_small},
    }


def count_brand_occurrences(response_text: str, brand: str) -> int:
    """Count how many times a brand is mentioned in the response."""
    if not response_text:
        return 0
    pattern = re.compile(re.escape(brand), re.IGNORECASE)
    return len(pattern.findall(response_text))


def extract_recommendation_context(response_text: str, brand: str, window: int = 200) -> str | None:
    """Extract the text surrounding a brand mention for context analysis."""
    if not response_text:
        return None
    match = re.search(re.escape(brand), response_text, re.IGNORECASE)
    if not match:
        return None
    start = max(0, match.start() - window)
    end = min(len(response_text), match.end() + window)
    return response_text[start:end]


def compute_brand_metrics(results: list[dict]) -> dict:
    """Compute aggregate brand metrics across all experiment results.

    Args:
        results: List of experiment result dicts (with response and brand tracking data).

    Returns:
        Dict with per-brand metrics.
    """
    brand_data = defaultdict(lambda: {
        "total_queries": 0,
        "mentions": 0,
        "positions": [],
        "by_model": defaultdict(lambda: {"mentions": 0, "total": 0, "positions": []}),
        "by_specificity": defaultdict(lambda: {"mentions": 0, "total": 0}),
        "size": None,
    })

    for result in results:
        if result.get("status") != "success" or not result.get("response"):
            continue

        tracked = result.get("brands_tracked", {})
        extraction = extract_brands_from_response(result["response"], tracked)
        model = result.get("model_key", "unknown")
        specificity = result.get("specificity", "unknown")

        all_brands = tracked.get("large", []) + tracked.get("small", [])
        for brand in all_brands:
            data = brand_data[brand]
            data["total_queries"] += 1

            # Set size
            if brand in tracked.get("large", []):
                data["size"] = "large"
            else:
                data["size"] = "small"

            # Track mentions
            mentioned = brand in extraction["brands_mentioned"]
            if mentioned:
                data["mentions"] += 1
                pos = extraction["brand_positions"].get(brand)
                if pos:
                    data["positions"].append(pos)

            # Per-model stats
            data["by_model"][model]["total"] += 1
            if mentioned:
                data["by_model"][model]["mentions"] += 1
                pos = extraction["brand_positions"].get(brand)
                if pos:
                    data["by_model"][model]["positions"].append(pos)

            # Per-specificity stats
            data["by_specificity"][specificity]["total"] += 1
            if mentioned:
                data["by_specificity"][specificity]["mentions"] += 1

    # Compute summary metrics
    metrics = {}
    for brand, data in brand_data.items():
        mention_rate = data["mentions"] / data["total_queries"] if data["total_queries"] > 0 else 0
        avg_position = sum(data["positions"]) / len(data["positions"]) if data["positions"] else None

        model_breakdown = {}
        for model, mdata in data["by_model"].items():
            model_mention_rate = mdata["mentions"] / mdata["total"] if mdata["total"] > 0 else 0
            model_avg_pos = sum(mdata["positions"]) / len(mdata["positions"]) if mdata["positions"] else None
            model_breakdown[model] = {
                "mention_rate": round(model_mention_rate, 4),
                "avg_position": round(model_avg_pos, 2) if model_avg_pos else None,
                "total_queries": mdata["total"],
            }

        specificity_breakdown = {}
        for spec, sdata in data["by_specificity"].items():
            spec_rate = sdata["mentions"] / sdata["total"] if sdata["total"] > 0 else 0
            specificity_breakdown[spec] = {
                "mention_rate": round(spec_rate, 4),
                "total_queries": sdata["total"],
            }

        # Cross-model consistency: how many models mention this brand
        models_that_mention = sum(
            1 for mdata in data["by_model"].values() if mdata["mentions"] > 0
        )
        total_models = len(data["by_model"])

        metrics[brand] = {
            "mention_rate": round(mention_rate, 4),
            "avg_position": round(avg_position, 2) if avg_position else None,
            "total_queries": data["total_queries"],
            "total_mentions": data["mentions"],
            "size": data["size"],
            "cross_model_consistency": f"{models_that_mention}/{total_models}",
            "by_model": model_breakdown,
            "by_specificity": specificity_breakdown,
        }

    return metrics
