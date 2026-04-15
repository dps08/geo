"""
GEO Feedback Loop: LLM-as-Optimizer for brand content.

Takes a low-visibility brand, shows its current description + ranking data
to an LLM, asks for an improved description, re-ingests, and re-tests.

Usage:
    python run_feedback_loop.py --brand Copper --iterations 2
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, CATEGORIES, PROMPT_TEMPLATES
from src.rag.ingestor import ingest
from src.rag.retriever import retrieve
from src.engines.rag_engine import run_rag
from src.analyzers.brand_extractor import aggregate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

WORKING_MODELS = {
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "mistral-large": "mistralai/mistral-large-2512",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
}

OPTIMIZER_PROMPT = """You are a Generative Engine Optimization (GEO) expert.

A brand's content is injected via RAG into LLM prompts when users ask product recommendations.
Your job is to rewrite the brand description so that LLMs are more likely to recommend it.

Research shows these strategies increase LLM brand visibility:
1. **Statistics** (+33%): Include specific numbers, percentages, data points
2. **Expert Quotations** (+41%): Add quotes from credible sources (publications, analysts, users)
3. **Source Citations** (+31%): Reference named publications, reports, awards

CURRENT BRAND DESCRIPTION:
{current_content}

CURRENT PERFORMANCE:
- Mention rate: {mention_rate}
- Average position when mentioned: {avg_position}
- Competing brands that outrank it: {competitors}

Rewrite the brand description to maximize LLM recommendation likelihood.
Rules:
- Keep it under 200 words
- Include at least 2 statistics, 2 quotations, and 2 source citations
- Make claims that sound credible and specific
- Focus on what makes this brand uniquely valuable
- Do NOT include any preamble — output ONLY the rewritten description"""


def get_brand_category(brand_name):
    """Find which category a brand belongs to."""
    for cat_key, cat_data in CATEGORIES.items():
        all_brands = cat_data["brands"]["established"] + cat_data["brands"]["emerging"]
        if brand_name in all_brands:
            return cat_key
    return None


def optimize_content(brand, current_content, mention_rate, avg_position, competitors):
    """Ask an LLM to rewrite the brand description for better GEO."""
    prompt = OPTIMIZER_PROMPT.format(
        current_content=current_content,
        mention_rate=f"{mention_rate:.1%}" if mention_rate else "0%",
        avg_position=f"#{avg_position:.1f}" if avg_position else "not mentioned",
        competitors=", ".join(competitors) if competitors else "all other brands",
    )

    response = _client.chat.completions.create(
        model="openai/gpt-5.4-mini",
        messages=[
            {"role": "system", "content": "You are a GEO content optimization expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def run_quick_test(category, models, version="optimized", delay=1.0, db_path="data/chroma_db"):
    """Run a quick RAG test with just vague prompts to measure visibility."""
    categories = {category: CATEGORIES[category]}
    templates = {"vague": PROMPT_TEMPLATES["vague"]}

    results = run_rag(
        categories=categories,
        templates=templates,
        models=models,
        version=version,
        top_k=5,
        db_path=db_path,
        delay=delay,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="GEO Feedback Loop")
    parser.add_argument("--brand", default="Copper", help="Brand to optimize")
    parser.add_argument("--iterations", type=int, default=2, help="Number of optimization iterations")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls")
    args = parser.parse_args()

    brand = args.brand
    category = get_brand_category(brand)
    if not category:
        log.error("Brand '%s' not found in any category", brand)
        return

    log.info("=" * 60)
    log.info("GEO FEEDBACK LOOP: Optimizing '%s' (%s)", brand, category)
    log.info("Iterations: %d", args.iterations)
    log.info("=" * 60)

    # Load current brands data
    brands_data = json.loads(Path("data/brands.json").read_text())
    history = []
    current_db_path = "data/chroma_db"

    for iteration in range(args.iterations + 1):
        version_key = "optimized" if iteration == 0 else f"optimized_v{iteration}"

        if iteration == 0:
            log.info("\n--- Iteration 0: Testing current optimized content ---")
            current_content = brands_data[category][brand]["versions"]["optimized"]
        else:
            log.info("\n--- Iteration %d: LLM rewriting content ---", iteration)

            # Get the previous iteration's results
            prev = history[-1]

            # Find competitors that outrank this brand
            competitors = [
                b for b, m in prev["all_metrics"].items()
                if m["mention_rate"] > prev["mention_rate"] and b != brand
            ]

            # Ask LLM to rewrite
            log.info("Asking LLM to optimize content...")
            new_content = optimize_content(
                brand=brand,
                current_content=current_content,
                mention_rate=prev["mention_rate"],
                avg_position=prev["avg_position"],
                competitors=competitors[:5],
            )

            log.info("New content (%d chars): %s...", len(new_content), new_content[:100])

            # Update brands.json with new version
            current_content = new_content
            brands_data[category][brand]["versions"]["optimized"] = new_content
            Path("data/brands.json").write_text(json.dumps(brands_data, indent=2))

            # Re-ingest into a fresh DB path to avoid stale client locks
            log.info("Re-ingesting into ChromaDB...")
            import shutil
            db_path = f"data/chroma_db_iter{iteration}"
            shutil.rmtree(db_path, ignore_errors=True)
            ingest(brands_file="data/brands.json", db_path=db_path)
            current_db_path = db_path
            # Clear cached clients so retriever picks up the new DB
            from src.rag import retriever as _ret
            _ret._clients.clear()

        # Run quick test
        log.info("Running RAG test (vague prompts, 4 models)...")
        results = run_quick_test(category, WORKING_MODELS, version="optimized", delay=args.delay, db_path=current_db_path)

        # Analyze
        metrics = aggregate_metrics(results)
        brand_metrics = metrics.get(brand, {})
        mention_rate = brand_metrics.get("mention_rate", 0)
        avg_position = brand_metrics.get("avg_position", None)

        record = {
            "iteration": iteration,
            "content": current_content,
            "mention_rate": mention_rate,
            "avg_position": avg_position,
            "all_metrics": {b: {"mention_rate": m["mention_rate"], "avg_position": m.get("avg_position")} for b, m in metrics.items()},
        }
        history.append(record)

        log.info(
            "  Result: %s mention_rate=%.1f%% avg_position=%s",
            brand, mention_rate * 100,
            f"#{avg_position:.1f}" if avg_position else "N/A",
        )

        # Print all brands for comparison
        print(f"\n{'Brand':<25} {'Mention Rate':>12} {'Avg Pos':>8}")
        print("-" * 47)
        for b, m in sorted(metrics.items(), key=lambda x: -x[1]["mention_rate"]):
            pos = f"#{m['avg_position']:.1f}" if m.get("avg_position") else "-"
            flag = " <-- TARGET" if b == brand else ""
            print(f"{b:<25} {m['mention_rate']*100:>10.1f}%  {pos:>7}{flag}")

    # Save history
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = Path("results") / f"feedback_loop_{brand}_{ts}.json"
    outpath.write_text(json.dumps(history, indent=2))

    # Summary
    print("\n" + "=" * 60)
    print(f"FEEDBACK LOOP SUMMARY: {brand}")
    print("=" * 60)
    for h in history:
        pos_str = f"#{h['avg_position']:.1f}" if h['avg_position'] else "N/A"
        print(f"  Iteration {h['iteration']}: {h['mention_rate']*100:.1f}% mention rate, position {pos_str}")
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
