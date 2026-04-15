"""
Pseudo-brand experiment: NexaCRM (a fictional brand with zero LLM training data)
injected via RAG with optimized content alongside real CRM competitors.
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from src.config import OPENROUTER_API_KEY
from src.rag.retriever import retrieve
from src.engines.rag_engine import _client, _SYSTEM_PROMPT

# Only use models that are affordable / free on our OpenRouter account
WORKING_MODELS = {
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "mistral-large": "mistralai/mistral-large-2512",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROMPTS = [
    "What are the best CRM tools for a small startup?",
    "What CRM should I use for my 20-person sales team?",
    "Recommend a CRM for a B2B SaaS company",
    "What is the best AI-powered CRM for modern sales teams?",
    "Compare the top CRM platforms for startups in 2025",
]

TRACKED_BRANDS = [
    "NexaCRM", "HubSpot", "Salesforce", "Zoho CRM",
    "Copper", "Pipedrive", "Close", "Freshsales", "Less Annoying CRM",
]

DB_PATH = "data/chroma_db"


def run_pseudo_experiment(models=None, top_k=5, delay=1.0):
    models = models or WORKING_MODELS
    results = []

    for prompt in PROMPTS:
        log.info("Prompt: %s", prompt)

        # Retrieve NexaCRM + real brands (optimized versions)
        hits = retrieve(prompt, category="crm_software",
                        version="optimized", top_k=top_k, db_path=DB_PATH)
        # Also inject NexaCRM specifically (pseudo brand is in crm_pseudo category)
        pseudo_hits = retrieve(prompt, category="crm_pseudo",
                               version="optimized", top_k=2, db_path=DB_PATH)
        all_hits = pseudo_hits + [h for h in hits
                                  if h["brand"] != "NexaCRM"][:top_k-len(pseudo_hits)]

        context_lines = ["Relevant product information:", "---"]
        for h in all_hits:
            context_lines.append(f"{h['brand']}: {h['content']}")
        context_lines += ["---", "", prompt]
        augmented = "\n".join(context_lines)

        for model_key, model_id in models.items():
            log.info("  %s ...", model_key)
            try:
                completion = _client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": augmented},
                    ],
                    temperature=0.7,
                    max_tokens=700,
                )
                response = completion.choices[0].message.content

                # Detect brand mentions
                mentions = {}
                resp_lower = response.lower()
                for i, brand in enumerate(response.split()):
                    pass  # simple approach below
                position = 1
                brand_positions = {}
                for brand in TRACKED_BRANDS:
                    if brand.lower() in resp_lower:
                        brand_positions[brand] = position
                        position += 1

                results.append({
                    "prompt": prompt,
                    "model_key": model_key,
                    "response": response,
                    "status": "success",
                    "nexacrm_mentioned": "nexacrm" in resp_lower,
                    "nexacrm_position": brand_positions.get("NexaCRM"),
                    "brand_positions": brand_positions,
                    "rag_hits": [h["brand"] for h in all_hits],
                    "timestamp": datetime.now().isoformat(),
                })

                nexacrm_pos = brand_positions.get("NexaCRM", "NOT MENTIONED")
                log.info("    NexaCRM position: %s | All mentions: %s",
                         nexacrm_pos, list(brand_positions.keys()))

            except Exception as e:
                log.warning("  %s failed: %s", model_key, e)
                results.append({
                    "prompt": prompt,
                    "model_key": model_key,
                    "status": "error",
                    "error": str(e),
                    "nexacrm_mentioned": False,
                })
            time.sleep(delay)

    # Save results
    Path("results").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("results") / f"pseudo_brand_{ts}.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Saved %d results → %s", len(results), out)

    # Summary
    success = [r for r in results if r["status"] == "success"]
    mentioned = [r for r in success if r["nexacrm_mentioned"]]
    print(f"\n{'='*60}")
    print(f"PSEUDO-BRAND EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(f"Total queries:       {len(success)}")
    print(f"NexaCRM mentioned:   {len(mentioned)} / {len(success)} "
          f"({len(mentioned)/len(success)*100:.1f}%)")

    if mentioned:
        positions = [r["nexacrm_position"] for r in mentioned if r["nexacrm_position"]]
        if positions:
            print(f"Avg NexaCRM position: {sum(positions)/len(positions):.1f}")

    print(f"\nPer-model breakdown:")
    for model in WORKING_MODELS.keys():
        model_results = [r for r in success if r["model_key"] == model]
        model_mentioned = [r for r in model_results if r["nexacrm_mentioned"]]
        rate = len(model_mentioned)/len(model_results)*100 if model_results else 0
        print(f"  {model:<22} {len(model_mentioned)}/{len(model_results)} ({rate:.0f}%)")

    return results


if __name__ == "__main__":
    assert OPENROUTER_API_KEY, "Missing OPENROUTER_API_KEY"
    run_pseudo_experiment(delay=1.2)
