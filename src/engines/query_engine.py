"""Query engine that sends prompts to multiple LLMs via OpenRouter."""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from openai import OpenAI

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODELS


client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)


def query_single_model(prompt: str, model_key: str, model_id: str) -> dict:
    """Send a single prompt to a single model and return the response."""
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful product recommendation assistant. "
                        "When asked about product recommendations, provide specific brand names, "
                        "explain why you recommend them, and rank them in order of preference. "
                        "Be specific about features, pricing, and use cases."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        return {
            "model_key": model_key,
            "model_id": model_id,
            "prompt": prompt,
            "response": response.choices[0].message.content,
            "timestamp": datetime.now().isoformat(),
            "tokens_used": {
                "prompt": response.usage.prompt_tokens if response.usage else 0,
                "completion": response.usage.completion_tokens if response.usage else 0,
            },
            "status": "success",
        }

    except Exception as e:
        return {
            "model_key": model_key,
            "model_id": model_id,
            "prompt": prompt,
            "response": None,
            "timestamp": datetime.now().isoformat(),
            "tokens_used": {"prompt": 0, "completion": 0},
            "status": "error",
            "error": str(e),
        }


def query_all_models(prompt: str, models: dict | None = None, delay: float = 1.0) -> list[dict]:
    """Send a prompt to all configured models and collect responses."""
    if models is None:
        models = MODELS

    results = []
    for model_key, model_id in models.items():
        print(f"  Querying {model_key}...", end=" ", flush=True)
        result = query_single_model(prompt, model_key, model_id)

        if result["status"] == "success":
            print("OK")
        else:
            print(f"FAILED: {result.get('error', 'unknown')}")

        results.append(result)
        time.sleep(delay)  # rate limiting

    return results


def generate_prompts(category_key: str, category_display_name: str, templates: dict) -> list[dict]:
    """Generate all prompts for a given category using the templates."""
    prompts = []
    for specificity, template_list in templates.items():
        for template in template_list:
            prompt_text = template.format(category=category_display_name.lower())
            prompt_id = hashlib.md5(f"{category_key}:{specificity}:{prompt_text}".encode()).hexdigest()[:12]
            prompts.append({
                "id": prompt_id,
                "category": category_key,
                "specificity": specificity,
                "prompt": prompt_text,
            })
    return prompts


def run_baseline_experiment(
    categories: dict,
    templates: dict,
    models: dict | None = None,
    output_dir: str = "results",
    delay: float = 1.0,
) -> list[dict]:
    """Run the full baseline experiment across all categories, prompts, and models."""
    if models is None:
        models = MODELS

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    all_results = []
    total_queries = 0

    for cat_key, cat_info in categories.items():
        prompts = generate_prompts(cat_key, cat_info["display_name"], templates)
        print(f"\n{'='*60}")
        print(f"Category: {cat_info['display_name']}")
        print(f"Prompts: {len(prompts)} | Models: {len(models)}")
        print(f"Total queries for this category: {len(prompts) * len(models)}")
        print(f"{'='*60}")

        for i, prompt_info in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] ({prompt_info['specificity']}) {prompt_info['prompt']}")
            responses = query_all_models(prompt_info["prompt"], models, delay=delay)

            for response in responses:
                result = {
                    **prompt_info,
                    **response,
                    "brands_tracked": {
                        "large": cat_info["brands"]["large"],
                        "small": cat_info["brands"]["small"],
                    },
                }
                all_results.append(result)
                total_queries += 1

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"baseline_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment complete! Total queries: {total_queries}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")

    return all_results
