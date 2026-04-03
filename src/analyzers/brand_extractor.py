"""
Detects which tracked brands appear in an LLM response, determines
their mention order, and aggregates visibility metrics across an
entire experiment run.
"""

import re
from collections import defaultdict
from typing import Optional


def extract_brands(response_text: str, tracked_brands: dict) -> dict:
    """
    Scan ``response_text`` for mentions of tracked brands and rank them
    by order of first appearance.

    Args:
        response_text: Raw text from an LLM response.
        tracked_brands: Mapping with ``"established"`` and ``"emerging"``
            lists of brand names.

    Returns:
        A dict with keys ``brands_mentioned``, ``brand_positions``,
        ``brand_count``, and ``mentions_by_tier``.
    """
    empty = {
        "brands_mentioned": [],
        "brand_positions": {},
        "brand_count": 0,
        "mentions_by_tier": {"established": [], "emerging": []},
    }
    if not response_text:
        return empty

    all_brands = (
        tracked_brands.get("established", []) + tracked_brands.get("emerging", [])
    )
    lowered = response_text.lower()

    # Map each detected brand to its character offset in the response.
    found: dict[str, int] = {}
    for brand in all_brands:
        for variant in _brand_patterns(brand):
            match = re.search(r"\b" + variant + r"\b", lowered)
            if match:
                found[brand] = match.start()
                break

    ranked = sorted(found.items(), key=lambda pair: pair[1])
    positions = {brand: rank for rank, (brand, _) in enumerate(ranked, start=1)}

    established_set = set(tracked_brands.get("established", []))
    emerging_set = set(tracked_brands.get("emerging", []))

    return {
        "brands_mentioned": [b for b, _ in ranked],
        "brand_positions": positions,
        "brand_count": len(found),
        "mentions_by_tier": {
            "established": [b for b in found if b in established_set],
            "emerging": [b for b in found if b in emerging_set],
        },
    }


def count_occurrences(response_text: str, brand: str) -> int:
    """Return the number of times ``brand`` appears (case-insensitive)."""
    if not response_text:
        return 0
    return len(re.findall(re.escape(brand), response_text, re.IGNORECASE))


def extract_context(
    response_text: str,
    brand: str,
    window: int = 200,
) -> Optional[str]:
    """
    Pull out the text surrounding a brand mention so downstream analysis
    (e.g. sentiment) has focused input instead of the full response.
    """
    if not response_text:
        return None
    match = re.search(re.escape(brand), response_text, re.IGNORECASE)
    if not match:
        return None
    lo = max(0, match.start() - window)
    hi = min(len(response_text), match.end() + window)
    return response_text[lo:hi]


def aggregate_metrics(results: list[dict]) -> dict:
    """
    Compute per-brand visibility metrics over an experiment result set.

    For each brand, returns overall mention rate, average position,
    cross-model consistency, and breakdowns by model and query specificity.
    """
    accumulator = defaultdict(lambda: {
        "total": 0,
        "mentions": 0,
        "positions": [],
        "by_model": defaultdict(lambda: {"mentions": 0, "total": 0, "positions": []}),
        "by_specificity": defaultdict(lambda: {"mentions": 0, "total": 0}),
        "tier": None,
    })

    for result in results:
        if result.get("status") != "success" or not result.get("response"):
            continue

        tracked = result.get("brands_tracked", {})
        extraction = extract_brands(result["response"], tracked)
        model = result.get("model_key", "unknown")
        spec = result.get("specificity", "unknown")

        all_brands = (
            tracked.get("established", []) + tracked.get("emerging", [])
        )
        for brand in all_brands:
            acc = accumulator[brand]
            acc["total"] += 1
            acc["tier"] = (
                "established" if brand in tracked.get("established", []) else "emerging"
            )

            mentioned = brand in extraction["brands_mentioned"]
            if mentioned:
                acc["mentions"] += 1
                pos = extraction["brand_positions"].get(brand)
                if pos is not None:
                    acc["positions"].append(pos)

            acc["by_model"][model]["total"] += 1
            if mentioned:
                acc["by_model"][model]["mentions"] += 1
                pos = extraction["brand_positions"].get(brand)
                if pos is not None:
                    acc["by_model"][model]["positions"].append(pos)

            acc["by_specificity"][spec]["total"] += 1
            if mentioned:
                acc["by_specificity"][spec]["mentions"] += 1

    metrics = {}
    for brand, acc in accumulator.items():
        mention_rate = acc["mentions"] / acc["total"] if acc["total"] else 0
        avg_pos = (
            sum(acc["positions"]) / len(acc["positions"])
            if acc["positions"]
            else None
        )

        model_stats = {}
        for model, m in acc["by_model"].items():
            rate = m["mentions"] / m["total"] if m["total"] else 0
            mpos = (
                sum(m["positions"]) / len(m["positions"]) if m["positions"] else None
            )
            model_stats[model] = {
                "mention_rate": round(rate, 4),
                "avg_position": round(mpos, 2) if mpos else None,
                "queries": m["total"],
            }

        spec_stats = {}
        for spec, s in acc["by_specificity"].items():
            rate = s["mentions"] / s["total"] if s["total"] else 0
            spec_stats[spec] = {"mention_rate": round(rate, 4), "queries": s["total"]}

        models_present = sum(1 for m in acc["by_model"].values() if m["mentions"] > 0)

        metrics[brand] = {
            "mention_rate": round(mention_rate, 4),
            "avg_position": round(avg_pos, 2) if avg_pos else None,
            "total_queries": acc["total"],
            "total_mentions": acc["mentions"],
            "tier": acc["tier"],
            "cross_model_consistency": f"{models_present}/{len(acc['by_model'])}",
            "by_model": model_stats,
            "by_specificity": spec_stats,
        }
    return metrics


# -- internal helpers --------------------------------------------------------

def _brand_patterns(brand: str) -> list[str]:
    """
    Generate regex-friendly search variants for a brand name.

    Handles dotted names (Monday.com -> monday com) and multi-word brands
    where the first word alone is a common reference.
    """
    lowered = brand.lower()
    patterns = [re.escape(lowered)]
    without_dots = lowered.replace(".", "")
    if without_dots != lowered:
        patterns.append(re.escape(without_dots))
    if " " in lowered:
        patterns.append(re.escape(lowered.split()[0]))
    return patterns
