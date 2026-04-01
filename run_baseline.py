"""Run the baseline experiment — query LLMs and analyze brand visibility."""

import json
import argparse
from pathlib import Path

from src.config import CATEGORIES, PROMPT_TEMPLATES, MODELS
from src.engines.query_engine import run_baseline_experiment
from src.analyzers.brand_extractor import compute_brand_metrics
from src.analyzers.sentiment_analyzer import batch_sentiment_analysis, compute_sentiment_metrics
from src.utils.results_reporter import print_brand_metrics, print_sentiment_metrics, save_metrics_report


def main():
    parser = argparse.ArgumentParser(description="Run GEO baseline experiment")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CATEGORIES.keys()),
        help="Categories to test (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        help="Models to query (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip sentiment analysis (faster, cheaper)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 2 models and vague prompts only",
    )
    args = parser.parse_args()

    # Filter categories and models
    categories = {k: v for k, v in CATEGORIES.items() if k in args.categories}
    models = {k: v for k, v in MODELS.items() if k in args.models}

    # Quick mode: only 2 models, only vague prompts
    if args.quick:
        models = {k: v for k, v in list(MODELS.items())[:2]}
        templates = {"vague": PROMPT_TEMPLATES["vague"][:3]}
    else:
        templates = PROMPT_TEMPLATES

    print("=" * 60)
    print("GEO BASELINE EXPERIMENT")
    print("=" * 60)
    print(f"Categories: {list(categories.keys())}")
    print(f"Models: {list(models.keys())}")
    print(f"Prompt types: {list(templates.keys())}")
    total_prompts = sum(len(t) for t in templates.values()) * len(categories)
    total_queries = total_prompts * len(models)
    print(f"Total prompts: {total_prompts}")
    print(f"Total API calls: {total_queries}")
    print(f"Estimated time: ~{total_queries * (args.delay + 2):.0f} seconds")
    print("=" * 60)

    input("\nPress Enter to start (or Ctrl+C to cancel)...")

    # Step 1: Run queries
    print("\n>>> STEP 1: Querying LLMs...")
    results = run_baseline_experiment(
        categories=categories,
        templates=templates,
        models=models,
        delay=args.delay,
    )

    # Step 2: Compute brand metrics
    print("\n>>> STEP 2: Computing brand metrics...")
    brand_metrics = compute_brand_metrics(results)
    print_brand_metrics(brand_metrics)

    # Step 3: Sentiment analysis (optional)
    sentiment_metrics = None
    if not args.skip_sentiment:
        print("\n>>> STEP 3: Running sentiment analysis (LLM-as-judge)...")
        enriched = batch_sentiment_analysis(results, delay=args.delay)
        sentiment_metrics = compute_sentiment_metrics(enriched)
        print_sentiment_metrics(sentiment_metrics)
    else:
        print("\n>>> STEP 3: Skipping sentiment analysis (--skip-sentiment)")

    # Step 4: Save report
    print("\n>>> STEP 4: Saving report...")
    save_metrics_report(brand_metrics, sentiment_metrics)

    print("\nDone! Check the results/ directory for output files.")


if __name__ == "__main__":
    main()
