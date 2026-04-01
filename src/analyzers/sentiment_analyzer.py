"""LLM-as-judge sentiment analysis for brand mentions in LLM responses."""

import json
import time
from openai import OpenAI

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, JUDGE_MODEL
from src.analyzers.brand_extractor import extract_recommendation_context


client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

JUDGE_SYSTEM_PROMPT = """You are an expert at analyzing product recommendations.
Given a text excerpt from an AI assistant's response about a specific brand/product,
evaluate the sentiment and recommendation strength.

Return ONLY valid JSON with these fields:
{
    "sentiment": "positive" | "neutral" | "negative",
    "sentiment_score": <float 0.0 to 1.0, where 1.0 is most positive>,
    "recommendation_strength": "strong" | "moderate" | "weak" | "against",
    "key_attributes": [<list of attributes mentioned about the brand, e.g. "easy to use", "expensive">],
    "is_primary_recommendation": <bool, true if this brand is the #1 recommendation>
}

Be objective. Base your analysis only on what the text says."""


def analyze_brand_sentiment(response_text: str, brand: str) -> dict:
    """Use LLM-as-judge to analyze sentiment toward a specific brand in a response.

    Args:
        response_text: Full LLM response text.
        brand: The brand name to analyze sentiment for.

    Returns:
        Dict with sentiment analysis results.
    """
    context = extract_recommendation_context(response_text, brand, window=300)
    if not context:
        return {
            "brand": brand,
            "sentiment": "not_mentioned",
            "sentiment_score": 0.0,
            "recommendation_strength": "none",
            "key_attributes": [],
            "is_primary_recommendation": False,
            "status": "not_found",
        }

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Analyze the sentiment toward '{brand}' in this excerpt:\n\n"
                        f'"""{context}"""'
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        result["brand"] = brand
        result["status"] = "success"
        return result

    except (json.JSONDecodeError, Exception) as e:
        return {
            "brand": brand,
            "sentiment": "unknown",
            "sentiment_score": 0.5,
            "recommendation_strength": "unknown",
            "key_attributes": [],
            "is_primary_recommendation": False,
            "status": "error",
            "error": str(e),
        }


def analyze_all_brands_in_response(response_text: str, tracked_brands: dict, delay: float = 0.5) -> list[dict]:
    """Analyze sentiment for all tracked brands mentioned in a response.

    Args:
        response_text: Full LLM response text.
        tracked_brands: Dict with 'large' and 'small' brand lists.
        delay: Delay between API calls for rate limiting.

    Returns:
        List of sentiment analysis results for each mentioned brand.
    """
    all_brands = tracked_brands.get("large", []) + tracked_brands.get("small", [])
    results = []

    for brand in all_brands:
        if brand.lower() in response_text.lower():
            result = analyze_brand_sentiment(response_text, brand)
            results.append(result)
            time.sleep(delay)

    return results


def batch_sentiment_analysis(experiment_results: list[dict], delay: float = 0.5) -> list[dict]:
    """Run sentiment analysis on all results from a baseline experiment.

    Args:
        experiment_results: List of experiment result dicts.
        delay: Delay between API calls.

    Returns:
        List of enriched results with sentiment data.
    """
    enriched = []
    total = len(experiment_results)

    for i, result in enumerate(experiment_results):
        if result.get("status") != "success" or not result.get("response"):
            enriched.append({**result, "sentiment_analysis": []})
            continue

        tracked = result.get("brands_tracked", {})
        print(f"  [{i+1}/{total}] Analyzing sentiment for {result.get('model_key')}...", flush=True)

        sentiments = analyze_all_brands_in_response(
            result["response"], tracked, delay=delay
        )
        enriched.append({**result, "sentiment_analysis": sentiments})

    return enriched


def compute_sentiment_metrics(enriched_results: list[dict]) -> dict:
    """Compute aggregate sentiment metrics per brand.

    Args:
        enriched_results: Results with sentiment_analysis field.

    Returns:
        Dict mapping brand -> sentiment metrics.
    """
    from collections import defaultdict

    brand_sentiments = defaultdict(lambda: {
        "scores": [],
        "sentiments": [],
        "strengths": [],
        "attributes": [],
        "primary_count": 0,
        "total_analyzed": 0,
    })

    for result in enriched_results:
        for sa in result.get("sentiment_analysis", []):
            if sa.get("status") != "success":
                continue
            brand = sa["brand"]
            data = brand_sentiments[brand]
            data["total_analyzed"] += 1
            data["scores"].append(sa.get("sentiment_score", 0.5))
            data["sentiments"].append(sa.get("sentiment", "neutral"))
            data["strengths"].append(sa.get("recommendation_strength", "unknown"))
            data["attributes"].extend(sa.get("key_attributes", []))
            if sa.get("is_primary_recommendation"):
                data["primary_count"] += 1

    metrics = {}
    for brand, data in brand_sentiments.items():
        avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
        sentiment_dist = {}
        for s in ["positive", "neutral", "negative"]:
            sentiment_dist[s] = data["sentiments"].count(s)

        strength_dist = {}
        for s in ["strong", "moderate", "weak", "against"]:
            strength_dist[s] = data["strengths"].count(s)

        # Most common attributes
        attr_counts = {}
        for attr in data["attributes"]:
            attr_lower = attr.lower().strip()
            attr_counts[attr_lower] = attr_counts.get(attr_lower, 0) + 1
        top_attributes = sorted(attr_counts.items(), key=lambda x: -x[1])[:10]

        metrics[brand] = {
            "avg_sentiment_score": round(avg_score, 4),
            "sentiment_distribution": sentiment_dist,
            "strength_distribution": strength_dist,
            "top_attributes": top_attributes,
            "primary_recommendation_count": data["primary_count"],
            "total_analyzed": data["total_analyzed"],
        }

    return metrics
