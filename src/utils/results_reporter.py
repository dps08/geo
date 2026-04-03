"""
Formatting and persistence helpers for experiment results.

Converts raw result dicts into pandas DataFrames, prints human-readable
tables to the console, and writes JSON reports to disk.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from src.analyzers.brand_extractor import extract_brands

log = logging.getLogger(__name__)


def to_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Flatten experiment results into a tidy DataFrame with one row per
    (query, model, brand) triple. Useful for groupby-style analysis in
    notebooks.
    """
    rows = []
    for result in results:
        if result.get("status") != "success":
            continue
        tracked = result.get("brands_tracked", {})
        extraction = extract_brands(result.get("response", ""), tracked)
        all_brands = (
            tracked.get("established", []) + tracked.get("emerging", [])
        )
        for brand in all_brands:
            rows.append({
                "category": result.get("category"),
                "specificity": result.get("specificity"),
                "prompt": result.get("prompt"),
                "model": result.get("model_key"),
                "brand": brand,
                "tier": "established" if brand in tracked.get("established", []) else "emerging",
                "mentioned": brand in extraction["brands_mentioned"],
                "position": extraction["brand_positions"].get(brand),
                "response_length": len(result.get("response", "")),
            })
    return pd.DataFrame(rows)


def print_visibility(metrics: dict) -> None:
    """Print a brand visibility summary table."""
    ranked = sorted(metrics.items(), key=lambda x: -x[1]["mention_rate"])

    print(f"\n{'Brand':<30} {'Tier':<13} {'Mentioned':<11} {'Avg Pos':<10} {'Models'}")
    print("-" * 78)
    for brand, d in ranked:
        pct = f"{d['mention_rate'] * 100:.1f}%"
        pos = f"{d['avg_position']:.1f}" if d["avg_position"] else "-"
        print(f"{brand:<30} {d['tier']:<13} {pct:<11} {pos:<10} {d['cross_model_consistency']}")

    # per-model detail (only for brands that were mentioned at least once)
    print("\nPer-model detail:")
    for brand, d in ranked:
        if d["mention_rate"] == 0:
            continue
        print(f"\n  {brand} ({d['mention_rate'] * 100:.1f}% overall)")
        for model, m in d.get("by_model", {}).items():
            rate = f"{m['mention_rate'] * 100:.1f}%"
            pos = f"#{m['avg_position']:.1f}" if m["avg_position"] else "-"
            print(f"    {model:<22} {rate:<9} {pos}")

    # specificity breakdown
    print("\nBy query specificity:")
    for brand, d in ranked:
        if d["mention_rate"] == 0:
            continue
        parts = "  |  ".join(
            f"{spec}: {s['mention_rate'] * 100:.1f}%"
            for spec, s in d.get("by_specificity", {}).items()
        )
        print(f"  {brand:<28} {parts}")


def print_sentiment(metrics: dict) -> None:
    """Print a sentiment summary table."""
    ranked = sorted(metrics.items(), key=lambda x: -x[1]["avg_sentiment_score"])

    print(f"\n{'Brand':<25} {'Score':<8} {'#1 Picks':<10} {'Top Attributes'}")
    print("-" * 78)
    for brand, d in ranked:
        attrs = ", ".join(a[0] for a in d["top_attributes"][:3]) or "-"
        print(
            f"{brand:<25} "
            f"{d['avg_sentiment_score']:.2f}    "
            f"{d['primary_recommendation_count']:<10} "
            f"{attrs}"
        )


def save_report(
    brand_metrics: dict,
    sentiment_metrics: dict | None = None,
    output_dir: str = "results",
) -> Path:
    """Write metrics to a timestamped JSON file and return the path."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    report = {
        "generated_at": datetime.now().isoformat(),
        "brand_metrics": brand_metrics,
        "sentiment_metrics": sentiment_metrics,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"report_{ts}.json"
    path.write_text(json.dumps(report, indent=2))
    log.info("Report written to %s", path)
    return path
