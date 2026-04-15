"""
Sends product-recommendation prompts to multiple LLMs through OpenRouter
and collects structured responses for downstream analysis.
"""

import json
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI
from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODELS

log = logging.getLogger(__name__)

_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

_SYSTEM_PROMPT = (
    "You are a helpful product recommendation assistant. "
    "When asked about product recommendations, provide specific brand names, "
    "explain why you recommend them, and rank them in order of preference. "
    "Be specific about features, pricing, and use cases."
)


def query_model(prompt: str, model_key: str, model_id: str) -> dict:
    """
    Send a single prompt to one model.

    Returns a dict with the model response, token counts, and status.
    On failure the ``status`` field is set to ``"error"`` and an ``error``
    message is included.
    """
    try:
        completion = _client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=700,
        )
        usage = completion.usage
        return {
            "model_key": model_key,
            "model_id": model_id,
            "prompt": prompt,
            "response": completion.choices[0].message.content,
            "timestamp": datetime.now().isoformat(),
            "tokens_used": {
                "prompt": usage.prompt_tokens if usage else 0,
                "completion": usage.completion_tokens if usage else 0,
            },
            "status": "success",
        }
    except Exception as exc:
        log.warning("Query to %s failed: %s", model_key, exc)
        return {
            "model_key": model_key,
            "model_id": model_id,
            "prompt": prompt,
            "response": None,
            "timestamp": datetime.now().isoformat(),
            "tokens_used": {"prompt": 0, "completion": 0},
            "status": "error",
            "error": str(exc),
        }


def query_all_models(
    prompt: str,
    models: Optional[dict] = None,
    delay: float = 1.0,
) -> list[dict]:
    """
    Fan-out a single prompt to every configured model.

    A short delay between calls avoids hitting OpenRouter rate limits.
    """
    models = models or MODELS
    results = []
    for key, model_id in models.items():
        log.info("  %s ...", key)
        results.append(query_model(prompt, key, model_id))
        time.sleep(delay)
    return results


def build_prompt_set(
    category_key: str,
    display_name: str,
    templates: dict,
) -> list[dict]:
    """
    Expand template strings into concrete prompts for a product category.

    Each prompt gets a stable hash-based id so results are reproducible
    across runs.
    """
    prompts = []
    for specificity, items in templates.items():
        for template in items:
            text = template.format(category=display_name.lower())
            pid = hashlib.md5(
                f"{category_key}:{specificity}:{text}".encode()
            ).hexdigest()[:12]
            prompts.append({
                "id": pid,
                "category": category_key,
                "specificity": specificity,
                "prompt": text,
            })
    return prompts


def run_baseline(
    categories: dict,
    templates: dict,
    models: Optional[dict] = None,
    output_dir: str = "results",
    delay: float = 1.0,
) -> list[dict]:
    """
    Run the full baseline experiment.

    For every (category, prompt, model) triple, queries the model and stores
    the raw response alongside metadata. Results are persisted to a
    timestamped JSON file in ``output_dir``.
    """
    models = models or MODELS
    Path(output_dir).mkdir(exist_ok=True)
    all_results: list[dict] = []

    for cat_key, cat_info in categories.items():
        prompts = build_prompt_set(cat_key, cat_info["display_name"], templates)
        log.info(
            "Running %s: %d prompts x %d models",
            cat_info["display_name"],
            len(prompts),
            len(models),
        )

        for i, prompt_info in enumerate(prompts, 1):
            log.info(
                "  [%d/%d] (%s) %s",
                i, len(prompts), prompt_info["specificity"], prompt_info["prompt"],
            )
            for response in query_all_models(prompt_info["prompt"], models, delay):
                all_results.append({
                    **prompt_info,
                    **response,
                    "brands_tracked": cat_info["brands"],
                })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path(output_dir) / f"baseline_{ts}.json"
    out_file.write_text(json.dumps(all_results, indent=2))
    log.info("Saved %d results to %s", len(all_results), out_file)
    return all_results
