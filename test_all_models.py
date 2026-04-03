"""
Sends one prompt to every configured model and prints a comparison
of which brands each model recommends and in what order.
"""

from src.config import MODELS
from src.engines.query_engine import query_all_models
from src.analyzers.brand_extractor import extract_brands

TRACKED = {
    "established": ["Salesforce", "HubSpot", "Zoho CRM", "Microsoft Dynamics 365"],
    "emerging": ["Copper", "Close", "Freshsales", "Less Annoying CRM", "Pipedrive"],
}

PROMPT = "What are the best CRM tools for a small startup with 20 employees?"


def main():
    print(f"Prompt: {PROMPT}\n")
    results = query_all_models(PROMPT, delay=1.5)

    print(f"{'Model':<22} {'Status':<9} {'Brands (ranked)'}")
    print("-" * 80)

    for r in results:
        name = r["model_key"]
        if r["status"] != "success":
            print(f"{name:<22} {'fail':<9} {r.get('error', '')[:50]}")
            continue
        ext = extract_brands(r["response"], TRACKED)
        ranked = ", ".join(
            f"{b} (#{ext['brand_positions'][b]})" for b in ext["brands_mentioned"]
        )
        print(f"{name:<22} {'ok':<9} {ranked}")


if __name__ == "__main__":
    main()
