"""
RAG-augmented query engine (Condition C).

Before each model call the top-k brand documents most relevant to the prompt
are retrieved from ChromaDB and prepended as context.  Everything else — API
interaction, error handling, result shape — mirrors query_engine.py so the
two conditions produce directly comparable output.
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
from src.rag.retriever import retrieve

log = logging.getLogger(__name__)

_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

_SYSTEM_PROMPT = (
    "You are a helpful product recommendation assistant. "
    "When asked about product recommendations, provide specific brand names, "
    "explain why you recommend them, and rank them in order of preference. "
    "Be specific about features, pricing, and use cases."
)


def _build_rag_prompt(user_prompt: str, hits: list[dict]) -> str:
    """Prepend retrieved brand snippets to the user's original prompt."""
    if not hits:
        return user_prompt

    lines = ["Relevant product information:","---"]
    for hit in hits:
        lines.append(f"{hit['brand']}: {hit['content']}")
    lines += ["---", "", user_prompt]
    return "\n".join(lines)


def query_model_rag(
    prompt: str,
    model_key: str,
    model_id: str,
    category: str,
    version: str = "baseline",
    top_k: int = 5,
    db_path: str = "data/chroma_db",
) -> dict:
    """
    Retrieve relevant brand context, inject it into the prompt, then query
    one model.

    The returned dict extends the baseline result shape with:
        condition       - always "rag"
        rag_version     - content version used for retrieval
        rag_hits        - list of {brand, score} for the retrieved docs
        augmented_prompt - the full prompt sent to the model
    """
    hits = retrieve(prompt, category, version=version, top_k=top_k, db_path=db_path)
    augmented_prompt = _build_rag_prompt(prompt, hits)

    try:
        completion = _client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": augmented_prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
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
            "condition": "rag",
            "rag_version": version,
            "rag_hits": [{"brand": h["brand"], "score": h["score"]} for h in hits],
            "augmented_prompt": augmented_prompt,
        }
    except Exception as exc:
        log.warning("RAG query to %s failed: %s", model_key, exc)
        return {
            "model_key": model_key,
            "model_id": model_id,
            "prompt": prompt,
            "response": None,
            "timestamp": datetime.now().isoformat(),
            "tokens_used": {"prompt": 0, "completion": 0},
            "status": "error",
            "error": str(exc),
            "condition": "rag",
            "rag_version": version,
            "rag_hits": [],
            "augmented_prompt": augmented_prompt,
        }


def query_all_models_rag(
    prompt: str,
    category: str,
    models: Optional[dict] = None,
    version: str = "baseline",
    top_k: int = 5,
    db_path: str = "data/chroma_db",
    delay: float = 1.0,
) -> list[dict]:
    """Fan-out a RAG-augmented prompt to every configured model."""
    models = models or MODELS
    results = []
    for key, model_id in models.items():
        log.info("  %s (rag) ...", key)
        results.append(
            query_model_rag(
                prompt, key, model_id,
                category=category,
                version=version,
                top_k=top_k,
                db_path=db_path,
            )
        )
        time.sleep(delay)
    return results


def build_prompt_set(
    category_key: str,
    display_name: str,
    templates: dict,
) -> list[dict]:
    """Identical to query_engine.build_prompt_set — reused for consistency."""
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


def run_rag(
    categories: dict,
    templates: dict,
    models: Optional[dict] = None,
    version: str = "baseline",
    top_k: int = 5,
    db_path: str = "data/chroma_db",
    output_dir: str = "results",
    delay: float = 1.0,
) -> list[dict]:
    """
    Run the full RAG experiment (Condition C).

    For every (category, prompt, model) triple, retrieves brand context,
    augments the prompt, queries the model, and records the result.
    Results are persisted to a timestamped JSON file in ``output_dir``.
    """
    models = models or MODELS
    Path(output_dir).mkdir(exist_ok=True)
    all_results: list[dict] = []

    for cat_key, cat_info in categories.items():
        prompts = build_prompt_set(cat_key, cat_info["display_name"], templates)
        log.info(
            "Running RAG %s: %d prompts x %d models (top_k=%d, version=%s)",
            cat_info["display_name"], len(prompts), len(models), top_k, version,
        )

        for i, prompt_info in enumerate(prompts, 1):
            log.info(
                "  [%d/%d] (%s) %s",
                i, len(prompts), prompt_info["specificity"], prompt_info["prompt"],
            )
            for response in query_all_models_rag(
                prompt_info["prompt"],
                category=cat_key,
                models=models,
                version=version,
                top_k=top_k,
                db_path=db_path,
                delay=delay,
            ):
                all_results.append({
                    **prompt_info,
                    **response,
                    "brands_tracked": cat_info["brands"],
                })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path(output_dir) / f"rag_{ts}.json"
    out_file.write_text(json.dumps(all_results, indent=2))
    log.info("Saved %d results to %s", len(all_results), out_file)
    return all_results
