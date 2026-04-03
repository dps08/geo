"""
Smoke test: verifies that querying, brand extraction, and sentiment
analysis all work end-to-end with a single prompt to a single model.
"""

from src.config import OPENROUTER_API_KEY
from src.engines.query_engine import query_model
from src.analyzers.brand_extractor import extract_brands
from src.analyzers.sentiment_analyzer import analyze_sentiment

TRACKED = {
    "established": ["Salesforce", "HubSpot", "Zoho CRM", "Microsoft Dynamics 365"],
    "emerging": ["Copper", "Close", "Freshsales", "Less Annoying CRM", "Pipedrive"],
}


def main():
    assert OPENROUTER_API_KEY, "Missing OPENROUTER_API_KEY in .env"
    print(f"API key: {OPENROUTER_API_KEY[:15]}...\n")

    # query
    result = query_model(
        "What are the top 3 CRM tools for startups?",
        model_key="gpt-5.4-mini",
        model_id="openai/gpt-5.4-mini",
    )
    assert result["status"] == "success", f"Query failed: {result.get('error')}"
    print(f"Response ({len(result['response'])} chars):\n{result['response'][:300]}...\n")

    # extraction
    extraction = extract_brands(result["response"], TRACKED)
    print(f"Brands detected: {extraction['brands_mentioned']}")
    print(f"Positions: {extraction['brand_positions']}")
    print(f"By tier: {extraction['mentions_by_tier']}\n")

    # sentiment
    if extraction["brands_mentioned"]:
        brand = extraction["brands_mentioned"][0]
        sent = analyze_sentiment(result["response"], brand)
        print(f"Sentiment for {brand}:")
        print(f"  score:      {sent['sentiment_score']}")
        print(f"  tone:       {sent['sentiment']}")
        print(f"  strength:   {sent['recommendation_strength']}")
        print(f"  attributes: {sent['key_attributes']}")
        print(f"  primary:    {sent['is_primary_recommendation']}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
