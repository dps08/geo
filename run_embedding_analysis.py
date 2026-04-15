"""
Embedding Space Analysis: Compare LLM responses across experimental conditions.

Embeds all successful responses from Conditions A, B, C using sentence-transformers,
then visualizes how the embedding clusters differ between conditions using t-SNE/PCA.

Usage:
    python run_embedding_analysis.py
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_results(pattern):
    """Load the most recent result file matching pattern."""
    files = sorted(RESULTS_DIR.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return []
    log.info("Loading %s", files[0])
    return json.loads(files[0].read_text())


def get_successful_responses(results):
    """Extract successful response texts with metadata."""
    entries = []
    for r in results:
        if r.get("status") == "success" and r.get("response"):
            entries.append({
                "response": r["response"],
                "category": r.get("category", "unknown"),
                "model": r.get("model_key", "unknown"),
                "prompt": r.get("prompt", "")[:50],
            })
    return entries


def main():
    # Load results from each condition
    log.info("Loading experiment results...")
    baseline_results = load_results("baseline_*.json")
    rag_baseline_files = sorted(RESULTS_DIR.glob("rag_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    # We need to distinguish RAG baseline vs RAG optimized by checking version in results
    rag_results_all = []
    for f in rag_baseline_files[:5]:  # check recent files
        data = json.loads(f.read_text())
        if data and len(data) > 0:
            rag_results_all.append((f, data))

    # Separate by version if available, otherwise use file order
    cond_a = get_successful_responses(baseline_results)

    # For RAG files, try to separate by version
    cond_b_responses = []
    cond_c_responses = []
    for f, data in rag_results_all:
        sample = data[0] if data else {}
        version = sample.get("version", "unknown")
        responses = get_successful_responses(data)
        if version == "optimized":
            cond_b_responses = responses
        elif version == "baseline":
            cond_c_responses = responses
        elif not cond_c_responses:
            cond_c_responses = responses
        elif not cond_b_responses:
            cond_b_responses = responses

    # Also load pseudo-brand results
    pseudo_results = load_results("pseudo_brand_*.json")
    pseudo_responses = get_successful_responses(pseudo_results)

    log.info("Responses: A=%d, B=%d, C=%d, Pseudo=%d",
             len(cond_a), len(cond_b_responses), len(cond_c_responses), len(pseudo_responses))

    # Combine all responses
    all_texts = []
    all_labels = []
    all_models = []

    for r in cond_a:
        all_texts.append(r["response"])
        all_labels.append("A: Baseline")
        all_models.append(r["model"])
    for r in cond_c_responses:
        all_texts.append(r["response"])
        all_labels.append("C: RAG+Neutral")
        all_models.append(r["model"])
    for r in cond_b_responses:
        all_texts.append(r["response"])
        all_labels.append("B: RAG+Optimized")
        all_models.append(r["model"])
    for r in pseudo_responses:
        all_texts.append(r["response"])
        all_labels.append("Pseudo: NexaCRM")
        all_models.append(r["model"])

    if len(all_texts) < 10:
        log.error("Not enough responses to analyze. Need at least 10, got %d", len(all_texts))
        return

    # Embed all responses
    log.info("Embedding %d responses with sentence-transformers...", len(all_texts))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(all_texts, show_progress_bar=True)

    # === FIGURE 5: t-SNE visualization by condition ===
    log.info("Computing t-SNE projection...")
    perplexity = min(30, len(all_texts) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(embeddings)

    colors = {
        "A: Baseline": "#546E7A",
        "C: RAG+Neutral": "#78909C",
        "B: RAG+Optimized": "#1565C0",
        "Pseudo: NexaCRM": "#C62828",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in ["A: Baseline", "C: RAG+Neutral", "B: RAG+Optimized", "Pseudo: NexaCRM"]:
        mask = [l == label for l in all_labels]
        if not any(mask):
            continue
        xs = coords[mask, 0]
        ys = coords[mask, 1]
        ax.scatter(xs, ys, c=colors[label], label=label, alpha=0.7, s=60, edgecolors="white", linewidth=0.5)

    ax.set_title("t-SNE of LLM Response Embeddings by Experimental Condition", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_embedding_tsne.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig5_embedding_tsne.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved fig5_embedding_tsne.pdf")

    # === FIGURE 6: t-SNE by model ===
    model_colors = {
        "gpt-5.4-mini": "#1565C0",
        "llama-4-maverick": "#2E7D32",
        "mistral-large": "#6A1B9A",
        "deepseek-v3.2": "#E65100",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    for m, color in model_colors.items():
        mask = [mod == m for mod in all_models]
        if not any(mask):
            continue
        xs = coords[mask, 0]
        ys = coords[mask, 1]
        ax.scatter(xs, ys, c=color, label=m, alpha=0.7, s=60, edgecolors="white", linewidth=0.5)

    ax.set_title("t-SNE of LLM Response Embeddings by Model", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_embedding_by_model.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig6_embedding_by_model.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved fig6_embedding_by_model.pdf")

    # === Compute inter-condition distances ===
    log.info("\n=== Embedding Distance Analysis ===")
    from scipy.spatial.distance import cdist

    for label_a, label_b in [
        ("A: Baseline", "C: RAG+Neutral"),
        ("A: Baseline", "B: RAG+Optimized"),
        ("C: RAG+Neutral", "B: RAG+Optimized"),
        ("A: Baseline", "Pseudo: NexaCRM"),
    ]:
        mask_a = np.array([l == label_a for l in all_labels])
        mask_b = np.array([l == label_b for l in all_labels])
        if not mask_a.any() or not mask_b.any():
            continue
        dists = cdist(embeddings[mask_a], embeddings[mask_b], metric="cosine")
        mean_dist = dists.mean()
        log.info("  %s ↔ %s: mean cosine distance = %.4f", label_a, label_b, mean_dist)

    # Intra-condition distances (coherence)
    for label in ["A: Baseline", "B: RAG+Optimized", "C: RAG+Neutral"]:
        mask = np.array([l == label for l in all_labels])
        if mask.sum() < 2:
            continue
        dists = cdist(embeddings[mask], embeddings[mask], metric="cosine")
        np.fill_diagonal(dists, np.nan)
        mean_dist = np.nanmean(dists)
        log.info("  %s internal coherence: mean cosine distance = %.4f", label, mean_dist)

    log.info("\nEmbedding analysis complete. Figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
