"""Quick test to verify the query engine works with a single prompt."""

from src.config import OPENROUTER_API_KEY
from src.engines.query_engine import query_single_model, query_all_models
from src.analyzers.brand_extractor import extract_brands_from_response
from src.analyzers.sentiment_analyzer import analyze_brand_sentiment

# Verify API key is loaded
print(f"API Key loaded: {'Yes' if OPENROUTER_API_KEY else 'NO - check .env file!'}")
print(f"Key prefix: {OPENROUTER_API_KEY[:20]}..." if OPENROUTER_API_KEY else "")

# Test with a single model first
print("\n--- Testing single model (gpt-4o-mini) ---")
result = query_single_model(
    prompt="What are the top 3 CRM tools for startups?",
    model_key="gpt-4o-mini",
    model_id="openai/gpt-4o-mini",
)

if result["status"] == "success":
    print(f"Status: SUCCESS")
    print(f"Response length: {len(result['response'])} chars")
    print(f"Tokens: {result['tokens_used']}")
    print(f"\nResponse preview:\n{result['response'][:500]}...")

    # Test brand extraction
    print("\n--- Testing brand extraction ---")
    tracked = {
        "large": ["Salesforce", "HubSpot", "Zoho CRM", "Microsoft Dynamics 365"],
        "small": ["Copper", "Close", "Freshsales", "Less Annoying CRM", "Pipedrive"],
    }
    extraction = extract_brands_from_response(result["response"], tracked)
    print(f"Brands found: {extraction['brands_mentioned']}")
    print(f"Positions: {extraction['brand_positions']}")
    print(f"By size: {extraction['mentions_by_size']}")

    # Test sentiment analysis on the first mentioned brand
    if extraction["brands_mentioned"]:
        first_brand = extraction["brands_mentioned"][0]
        print(f"\n--- Testing sentiment analysis for '{first_brand}' ---")
        sentiment = analyze_brand_sentiment(result["response"], first_brand)
        print(f"Sentiment: {sentiment.get('sentiment')}")
        print(f"Score: {sentiment.get('sentiment_score')}")
        print(f"Strength: {sentiment.get('recommendation_strength')}")
        print(f"Attributes: {sentiment.get('key_attributes')}")
        print(f"Primary rec: {sentiment.get('is_primary_recommendation')}")
else:
    print(f"Status: FAILED")
    print(f"Error: {result.get('error')}")

print("\n--- Quick test complete! ---")
