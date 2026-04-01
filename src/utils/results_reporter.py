"""Generate reports and visualizations from experiment results."""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert raw experiment results to a pandas DataFrame for analysis."""
    rows = []
    for result in results:
        if result.get("status") != "success":
            continue

        from src.analyzers.brand_extractor import extract_brands_from_response

        tracked = result.get("brands_tracked", {})
        extraction = extract_brands_from_response(result.get("response", ""), tracked)

        all_brands = tracked.get("large", []) + tracked.get("small", [])
        for brand in all_brands:
            mentioned = brand in extraction["brands_mentioned"]
            position = extraction["brand_positions"].get(brand)
            size = "large" if brand in tracked.get("large", []) else "small"

            rows.append({
                "category": result.get("category"),
                "specificity": result.get("specificity"),
                "prompt": result.get("prompt"),
                "model": result.get("model_key"),
                "brand": brand,
                "brand_size": size,
                "mentioned": mentioned,
                "position": position,
                "response_length": len(result.get("response", "")),
            })

    return pd.DataFrame(rows)


def print_brand_metrics(metrics: dict):
    """Pretty-print brand metrics to console."""
    print(f"\n{'='*80}")
    print(f"{'BRAND VISIBILITY METRICS':^80}")
    print(f"{'='*80}")

    # Sort by mention rate descending
    sorted_brands = sorted(metrics.items(), key=lambda x: -x[1]["mention_rate"])

    print(f"\n{'Brand':<30} {'Size':<8} {'Mention%':<10} {'Avg Pos':<10} {'Models':<10}")
    print(f"{'-'*68}")

    for brand, data in sorted_brands:
        mention_pct = f"{data['mention_rate']*100:.1f}%"
        avg_pos = f"{data['avg_position']:.1f}" if data['avg_position'] else "N/A"
        consistency = data.get('cross_model_consistency', 'N/A')

        print(f"{brand:<30} {data['size']:<8} {mention_pct:<10} {avg_pos:<10} {consistency:<10}")

    # Print per-model breakdown
    print(f"\n{'='*80}")
    print(f"{'PER-MODEL BREAKDOWN':^80}")
    print(f"{'='*80}")

    for brand, data in sorted_brands:
        if data["mention_rate"] == 0:
            continue
        print(f"\n  {brand} (overall: {data['mention_rate']*100:.1f}%)")
        for model, mdata in data.get("by_model", {}).items():
            rate = f"{mdata['mention_rate']*100:.1f}%"
            pos = f"pos {mdata['avg_position']:.1f}" if mdata['avg_position'] else "no pos"
            print(f"    {model:<20} {rate:<10} {pos}")

    # Print specificity analysis
    print(f"\n{'='*80}")
    print(f"{'QUERY SPECIFICITY ANALYSIS':^80}")
    print(f"{'='*80}")

    for brand, data in sorted_brands:
        if data["mention_rate"] == 0:
            continue
        print(f"\n  {brand}")
        for spec, sdata in data.get("by_specificity", {}).items():
            rate = f"{sdata['mention_rate']*100:.1f}%"
            print(f"    {spec:<12} {rate}")


def print_sentiment_metrics(sentiment_metrics: dict):
    """Pretty-print sentiment metrics to console."""
    print(f"\n{'='*80}")
    print(f"{'BRAND SENTIMENT ANALYSIS':^80}")
    print(f"{'='*80}")

    sorted_brands = sorted(sentiment_metrics.items(), key=lambda x: -x[1]["avg_sentiment_score"])

    print(f"\n{'Brand':<25} {'Score':<8} {'Primary#':<10} {'Top Attributes'}")
    print(f"{'-'*80}")

    for brand, data in sorted_brands:
        score = f"{data['avg_sentiment_score']:.2f}"
        primary = str(data['primary_recommendation_count'])
        attrs = ", ".join([a[0] for a in data['top_attributes'][:3]]) if data['top_attributes'] else "N/A"
        print(f"{brand:<25} {score:<8} {primary:<10} {attrs}")


def save_metrics_report(
    brand_metrics: dict,
    sentiment_metrics: dict | None = None,
    output_dir: str = "results",
):
    """Save metrics to a JSON report file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "brand_metrics": brand_metrics,
        "sentiment_metrics": sentiment_metrics,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"report_{timestamp}.json"

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_file}")
    return report_file
