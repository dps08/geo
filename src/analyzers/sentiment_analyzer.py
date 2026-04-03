"""
Uses an LLM-as-judge approach to evaluate how positively or negatively
each model talks about a brand. A smaller, cheaper model reads the
context around a brand mention and returns structured sentiment data.
"""

import json
import time
import logging
from collections import defaultdict
from typing import Optional

from openai import OpenAI
from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, JUDGE_MODEL
from src.analyzers.brand_extractor import extract_context

log = logging.getLogger(__name__)

_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

_JUDGE_PROMPT = (
    "You are an expert at analyzing product recommendations. "
    "Given a text excerpt about a specific brand, evaluate the sentiment "
    "and recommendation strength.\n\n"
    "Return ONLY valid JSON:\n"
    "{\n"
    '  "sentiment": "positive" | "neutral" | "negative",\n'
    '  "sentiment_score": <float 0.0-1.0>,\n'
    '  "recommendation_strength": "strong" | "moderate" | "weak" | "against",\n'
    '  "key_attributes": [<list of mentioned attributes>],\n'
    '  "is_primary_recommendation": <bool>\n'
    "}\n\n"
    "Base your analysis only on what the text says."
)

_NOT_MENTIONED = {
    "sentiment": "not_mentioned",
    "sentiment_score": 0.0,
    "recommendation_strength": "none",
    "key_attributes": [],
    "is_primary_recommendation": False,
}


def analyze_sentiment(response_text: str, brand: str) -> dict:
    """
    Evaluate how a single brand is portrayed in a given LLM response.

    Extracts the surrounding context, sends it to the judge model, and
    parses the JSON output into a structured result.
    """
    context = extract_context(response_text, brand, window=300)
    if not context:
        return {**_NOT_MENTIONED, "brand": brand, "status": "not_found"}

    try:
        completion = _client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": f"Analyze the sentiment toward '{brand}':\n\n\"{context}\"",
                },
            ],
            temperature=0.0,
            max_tokens=300,
        )
        parsed = _parse_json(completion.choices[0].message.content)
        parsed["brand"] = brand
        parsed["status"] = "success"
        return parsed

    except Exception as exc:
        log.warning("Sentiment analysis failed for %s: %s", brand, exc)
        return {
            "brand": brand,
            "sentiment": "unknown",
            "sentiment_score": 0.5,
            "recommendation_strength": "unknown",
            "key_attributes": [],
            "is_primary_recommendation": False,
            "status": "error",
            "error": str(exc),
        }


def analyze_response(
    response_text: str,
    tracked_brands: dict,
    delay: float = 0.5,
) -> list[dict]:
    """
    Run sentiment analysis on every tracked brand that appears in the response.
    """
    all_brands = (
        tracked_brands.get("established", []) + tracked_brands.get("emerging", [])
    )
    results = []
    for brand in all_brands:
        if brand.lower() in response_text.lower():
            results.append(analyze_sentiment(response_text, brand))
            time.sleep(delay)
    return results


def enrich_results(
    experiment_results: list[dict],
    delay: float = 0.5,
) -> list[dict]:
    """
    Attach sentiment analysis to every result in an experiment run.

    Each result dict gets a new ``sentiment_analysis`` key containing
    per-brand sentiment data.
    """
    enriched = []
    total = len(experiment_results)
    for i, result in enumerate(experiment_results, 1):
        if result.get("status") != "success" or not result.get("response"):
            enriched.append({**result, "sentiment_analysis": []})
            continue

        log.info("  [%d/%d] sentiment for %s", i, total, result.get("model_key"))
        sentiments = analyze_response(
            result["response"],
            result.get("brands_tracked", {}),
            delay=delay,
        )
        enriched.append({**result, "sentiment_analysis": sentiments})
    return enriched


def aggregate_sentiment(enriched_results: list[dict]) -> dict:
    """
    Roll up per-response sentiment into brand-level averages.

    Returns a dict keyed by brand name with average score, sentiment
    distribution, strength distribution, top-mentioned attributes, and
    how often the brand was the primary recommendation.
    """
    acc = defaultdict(lambda: {
        "scores": [],
        "sentiments": [],
        "strengths": [],
        "attributes": [],
        "primary_count": 0,
        "total": 0,
    })

    for result in enriched_results:
        for sa in result.get("sentiment_analysis", []):
            if sa.get("status") != "success":
                continue
            bucket = acc[sa["brand"]]
            bucket["total"] += 1
            bucket["scores"].append(sa.get("sentiment_score", 0.5))
            bucket["sentiments"].append(sa.get("sentiment", "neutral"))
            bucket["strengths"].append(sa.get("recommendation_strength", "unknown"))
            bucket["attributes"].extend(sa.get("key_attributes", []))
            if sa.get("is_primary_recommendation"):
                bucket["primary_count"] += 1

    metrics = {}
    for brand, data in acc.items():
        avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0

        attr_freq: dict[str, int] = {}
        for attr in data["attributes"]:
            key = attr.lower().strip()
            attr_freq[key] = attr_freq.get(key, 0) + 1
        top_attrs = sorted(attr_freq.items(), key=lambda x: -x[1])[:10]

        metrics[brand] = {
            "avg_sentiment_score": round(avg, 4),
            "sentiment_distribution": {
                s: data["sentiments"].count(s)
                for s in ("positive", "neutral", "negative")
            },
            "strength_distribution": {
                s: data["strengths"].count(s)
                for s in ("strong", "moderate", "weak", "against")
            },
            "top_attributes": top_attrs,
            "primary_recommendation_count": data["primary_count"],
            "total_analyzed": data["total"],
        }
    return metrics


def _parse_json(raw: str) -> dict:
    """Strip optional markdown fences and parse JSON."""
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)
