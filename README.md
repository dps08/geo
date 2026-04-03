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

## Project structure

```
src/
  config.py                  — models, categories, prompt templates
  engines/
    query_engine.py          — OpenRouter client, prompt fan-out, experiment runner
  analyzers/
    brand_extractor.py       — mention detection, position ranking, visibility metrics
    sentiment_analyzer.py    — LLM-as-judge sentiment evaluation
  utils/
    results_reporter.py      — DataFrame conversion, console tables, JSON export
run_baseline.py              — CLI entry point for running experiments
test_quick.py                — single-model smoke test
test_all_models.py           — cross-model comparison
```

## Key dependencies

- **openai** — OpenRouter uses an OpenAI-compatible API
- **spacy** — NLP utilities
- **pandas** — tabular analysis
- **scikit-learn** — metrics and clustering
- **chromadb** — vector store for RAG experiments (upcoming)
- **streamlit** — dashboard (upcoming)

## Research context

This project investigates three questions:

1. Do the content optimization strategies from the KDD 2024 GEO paper (quotation addition, statistics injection, source citations) generalize across current-generation LLMs?
2. Does RAG-augmented context injection improve brand visibility more than the model's parametric knowledge alone?
3. How does query specificity (vague vs. detailed prompts) influence which brands surface?

## License

MIT
