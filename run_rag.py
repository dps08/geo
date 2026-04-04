"""
Entry point for the RAG experiment (Condition C).

Queries multiple LLMs with RAG-augmented product-recommendation prompts,
extracts brand mentions, optionally runs sentiment analysis, and saves a
metrics report.  Mirrors run_baseline.py — results are directly comparable.

Usage:
    python run_rag.py                            # full run, all models
    python run_rag.py --quick --skip-sentiment   # fast validation
    python run_rag.py --top-k 3 --version baseline
"""

import argparse
import logging

from src.config import CATEGORIES, PROMPT_TEMPLATES, MODELS
from src.engines.rag_engine import run_rag
from src.analyzers.brand_extractor import aggregate_metrics
from src.analyzers.sentiment_analyzer import enrich_results, aggregate_sentiment
from src.utils.results_reporter import print_visibility, print_sentiment, save_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GEO RAG experiment (Condition C)")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES.keys()))
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a minimal subset (2 models, 3 prompts) for quick validation",
    )
    parser.add_argument(
        "--version", default="baseline",
        help="Brand content version to use for retrieval (default: baseline)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of brand documents to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--db-path", default="data/chroma_db",
        help="Path to the ChromaDB vector store (default: data/chroma_db)",
    )
    args = parser.parse_args()

    categories = {k: v for k, v in CATEGORIES.items() if k in args.categories}
    models = {k: v for k, v in MODELS.items() if k in args.models}

    if args.quick:
        models = dict(list(MODELS.items())[:2])
        templates = {"vague": PROMPT_TEMPLATES["vague"][:3]}
    else:
        templates = PROMPT_TEMPLATES

    n_prompts = sum(len(t) for t in templates.values()) * len(categories)
    n_calls = n_prompts * len(models)
    log.info(
        "Plan: %d prompts x %d models = %d API calls  [version=%s, top_k=%d]",
        n_prompts, len(models), n_calls, args.version, args.top_k,
    )

    input("\nPress Enter to begin (Ctrl+C to abort)... ")

    # 1 - collect RAG-augmented responses
    results = run_rag(
        categories=categories,
        templates=templates,
        models=models,
        version=args.version,
        top_k=args.top_k,
        db_path=args.db_path,
        delay=args.delay,
    )

    # 2 - brand visibility
    brand_metrics = aggregate_metrics(results)
    print_visibility(brand_metrics)

    # 3 - sentiment (optional, costs extra tokens)
    sentiment_metrics = None
    if not args.skip_sentiment:
        enriched = enrich_results(results, delay=args.delay)
        sentiment_metrics = aggregate_sentiment(enriched)
        print_sentiment(sentiment_metrics)

    # 4 - persist
    save_report(brand_metrics, sentiment_metrics)
    log.info("Done.")


if __name__ == "__main__":
    main()
