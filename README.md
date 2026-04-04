# GEO — Generative Engine Optimization

A research platform for measuring and improving brand visibility in LLM-generated product recommendations.

When users ask ChatGPT, Claude, or Gemini "what's the best CRM?", certain brands dominate while others never appear. This project studies what drives those recommendations and whether targeted content optimization can shift them — building on the foundational GEO research by [Aggarwal et al. (KDD 2024)](https://arxiv.org/abs/2311.09735).

## What it does

1. **Queries 7 LLMs** (GPT-5.4, Claude Sonnet 4.6, Gemini 3.1 Pro, Llama 4 Maverick, Mistral Large, DeepSeek V3.2) with product recommendation prompts across multiple specificity levels.
2. **Extracts brand mentions** from each response — which brands appear, in what order, and how consistently across models.
3. **Evaluates sentiment** using an LLM-as-judge approach — whether a brand is recommended strongly, moderately, or dismissed.
4. **Aggregates visibility metrics** — mention rate, average position, cross-model consistency, and breakdowns by query specificity.

## Setup

```bash
# clone and create a virtual environment
git clone https://github.com/dps08/geo.git
cd geo
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# configure API access
cp .env.example .env
# edit .env and add your OpenRouter API key
```

## Usage

```bash
# smoke test — single model, single prompt (~15 seconds)
python test_quick.py

# cross-model comparison — all 7 models, one prompt (~30 seconds)
python test_all_models.py

# quick experiment — 2 models, 3 prompts per category (~2 minutes)
python run_baseline.py --quick --skip-sentiment

# full baseline — all models, all prompts, with sentiment (~30 minutes)
python run_baseline.py

# target a single category
python run_baseline.py --categories crm_software

# use specific models only
python run_baseline.py --models gpt-5.4 claude-sonnet-4.6 gemini-3.1-pro
```

Results are written to `results/` as timestamped JSON files.

## RAG Implementation (Condition C)

The RAG pipeline implements Condition C of the experiment — brand content is retrieved from a local vector store and injected into each prompt before querying the models. This lets us measure whether retrieval-augmented context boosts brand visibility compared to the baseline (Condition A).

### What was added

- **`data/brands.json`** — single source of truth for all 18 brand descriptions across CRM Software and Project Management categories. Each brand entry supports multiple named versions (e.g. `baseline`, `optimized`) so different content strategies can be compared without separate files.
- **`src/rag/ingestor.py`** — reads `brands.json`, embeds each document using `sentence-transformers` (`all-MiniLM-L6-v2`), and upserts into a persistent ChromaDB collection. Re-running is safe — existing documents are overwritten, not duplicated.
- **`src/rag/retriever.py`** — given a query, category, and version, returns the top-k most semantically relevant brand documents. Results are filtered by category so CRM content never surfaces in project management queries.
- **`src/engines/rag_engine.py`** — mirrors `query_engine.py` but retrieves brand context first and prepends it to the user prompt before each model call. Every result is tagged with `"condition": "rag"` and includes the list of retrieved brands and their scores.
- **`ingest.py`** — one-time CLI script to embed brand content and populate ChromaDB.
- **`run_rag.py`** — experiment runner that mirrors `run_baseline.py`, with added `--version`, `--top-k`, and `--db-path` flags.
- **`test_rag.py`** — smoke test that ingests into a temporary DB, runs retrieval, queries one model, and validates brand extraction end-to-end.

### RAG prompt format

For each query, the top-k retrieved brand documents are prepended to the original prompt:

```
Relevant product information:
---
HubSpot: HubSpot is an all-in-one CRM platform...
Salesforce: Salesforce is the world's leading CRM...
---

Best CRM for a 50-person startup  ← original prompt
```

### How to use

```bash
# Step 1 — populate the vector store (run once, or after editing brands.json)
python ingest.py

# Step 2 — smoke test (~20 seconds)
python test_rag.py

# Step 3 — quick experiment run (2 models, 3 prompts)
python run_rag.py --quick --skip-sentiment

# Full run — all models, all prompts, with sentiment
python run_rag.py

# Adjust retrieval depth
python run_rag.py --top-k 3

# Target a specific category or models
python run_rag.py --categories crm_software --models gpt-5.4 claude-sonnet-4.6
```

RAG results are saved to `results/rag_TIMESTAMP.json` alongside baseline results in `results/baseline_TIMESTAMP.json` for direct comparison.

### Adding or updating brand content

Open `data/brands.json` and edit the text under `versions.baseline` for any brand. After saving, re-run `python ingest.py` to refresh the vector store. To add an optimized version for Condition B, add an `"optimized"` key alongside `"baseline"` and run with `--version optimized`.

## Project structure

```
src/
  config.py                  — models, categories, prompt templates
  engines/
    query_engine.py          — OpenRouter client, prompt fan-out, experiment runner (Condition A)
    rag_engine.py            — RAG-augmented query engine (Condition C)
  analyzers/
    brand_extractor.py       — mention detection, position ranking, visibility metrics
    sentiment_analyzer.py    — LLM-as-judge sentiment evaluation
  rag/
    ingestor.py              — embed brand docs and store in ChromaDB
    retriever.py             — semantic top-k retrieval filtered by category and version
  utils/
    results_reporter.py      — DataFrame conversion, console tables, JSON export
data/
  brands.json                — brand content for all categories and versions
run_baseline.py              — CLI entry point for baseline experiment (Condition A)
run_rag.py                   — CLI entry point for RAG experiment (Condition C)
ingest.py                    — one-time vector store population script
test_quick.py                — baseline smoke test
test_all_models.py           — cross-model comparison
test_rag.py                  — RAG pipeline smoke test
```

## Key dependencies

- **openai** — OpenRouter uses an OpenAI-compatible API
- **spacy** — NLP utilities
- **pandas** — tabular analysis
- **scikit-learn** — metrics and clustering
- **chromadb** — vector store for RAG retrieval
- **sentence-transformers** — document and query embedding (`all-MiniLM-L6-v2`)

## Research context

This project investigates three questions:

1. Do the content optimization strategies from the KDD 2024 GEO paper (quotation addition, statistics injection, source citations) generalize across current-generation LLMs?
2. Does RAG-augmented context injection improve brand visibility more than the model's parametric knowledge alone?
3. How does query specificity (vague vs. detailed prompts) influence which brands surface?

## License

MIT
