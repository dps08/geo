"""Test querying all models with a single prompt."""

from src.engines.query_engine import query_all_models
from src.analyzers.brand_extractor import extract_brands_from_response
from src.config import MODELS

prompt = "What are the best CRM tools for a small startup with 20 employees?"

tracked = {
    "large": ["Salesforce", "HubSpot", "Zoho CRM", "Microsoft Dynamics 365"],
    "small": ["Copper", "Close", "Freshsales", "Less Annoying CRM", "Pipedrive"],
}

print(f"Prompt: {prompt}")
print(f"Models: {list(MODELS.keys())}")
print(f"Tracking brands: {tracked['large'] + tracked['small']}")
print("=" * 60)

results = query_all_models(prompt, delay=1.5)

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

for result in results:
    model = result["model_key"]
    if result["status"] != "success":
        print(f"\n{model}: FAILED - {result.get('error', 'unknown')}")
        continue

    extraction = extract_brands_from_response(result["response"], tracked)
    print(f"\n{model}:")
    print(f"  Brands: {extraction['brands_mentioned']}")
    print(f"  Positions: {extraction['brand_positions']}")
    print(f"  Response preview: {result['response'][:150]}...")
