# GEO - Generative Engine Optimization

A research platform for measuring and improving brand visibility in LLM-generated product recommendations.

When users ask ChatGPT or Gemini "what's the best CRM?", certain brands dominate while others never appear. This project studies what drives those recommendations and whether targeted content optimization can shift them, building on the foundational GEO research by [Aggarwal et al. (KDD 2024)](https://arxiv.org/abs/2311.09735).

## What it does

1. **Queries 4 LLMs** (GPT-5.4-mini, Llama 4 Maverick, Mistral Large, DeepSeek V3.2) with product recommendation prompts across three specificity levels.
2. **Extracts brand mentions** from each response, tracking which brands appear, in what order, and how consistently across models.
3. **Evaluates sentiment** using an LLM-as-judge approach to classify whether a brand is recommended strongly, moderately, or dismissed.
4. **Aggregates visibility metrics** including mention rate, average position, cross-model consistency, and breakdowns by query specificity.

## Setup

```bash
git clone https://github.com/dps08/geo.git
cd geo
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and add your OpenRouter API key
```

## Usage

### Baseline experiment (Condition A)

```bash
# smoke test, single model, single prompt
python test_quick.py

# cross-model comparison, all models, one prompt
python test_all_models.py

# quick experiment, 2 models, 3 prompts per category
python run_baseline.py --quick --skip-sentiment

# full baseline, all models, all prompts, with sentiment
python run_baseline.py
```

### RAG experiment (Conditions B and C)

```bash
# populate the vector store (run once, or after editing brands.json)
python ingest.py

# smoke test
python test_rag.py

# quick run (2 models, 3 prompts)
python run_rag.py --quick --skip-sentiment

# full run with optimized content (Condition B)
python run_rag.py --version optimized

# full run with neutral content (Condition C)
python run_rag.py --version baseline
```

### Pseudo-brand experiment (NexaCRM)

```bash
python run_pseudo.py
```

### Feedback loop (LLM-as-optimizer)

```bash
python run_feedback_loop.py --brand Copper --iterations 2
```

### Embedding analysis

```bash
python run_embedding_analysis.py
```

Results are written to `results/` as timestamped JSON files.

## Streamlit Dashboard

An interactive dashboard for exploring all experimental results.

```bash
streamlit run app.py
```

The dashboard includes seven pages:

- **Command Center** - overview of all experiments with key metrics and takeaways
- **Condition Deep-Dive** - comparison across all three conditions with brand filtering, specificity analysis, and heatmaps
- **Model Intelligence** - per-model breakdown showing how each LLM responds differently to the same content
- **NexaCRM Experiment** - results from the pseudo-brand experiment with per-model hit rates and individual query results
- **Feedback Loop** - visualization of the closed-loop optimization process with content evolution
- **Embedding Analysis** - t-SNE projections and cosine distance matrices showing how RAG standardizes model behavior
- **Response Explorer** - browse raw LLM outputs from every experiment with filtering by brand and model

Supports dark and light mode toggle in the sidebar.

## Experimental conditions

- **Condition A (Baseline):** LLMs are queried with no extra context, relying only on parametric knowledge.
- **Condition B (RAG + Optimized):** Brand descriptions enriched with statistics, expert quotations, and source citations are injected via RAG.
- **Condition C (RAG + Neutral):** Plain factual brand descriptions are injected via RAG, isolating the effect of content quality from content presence.

## RAG pipeline

The pipeline uses sentence-transformers (all-MiniLM-L6-v2) to embed brand descriptions into ChromaDB. For each query, the top-k most relevant documents are retrieved and prepended to the prompt before sending to the LLM.

Prompt format:

```
Relevant product information:
---
HubSpot: HubSpot is an all-in-one CRM platform...
Salesforce: Salesforce is the world's leading CRM...
---

Best CRM for a 50-person startup
```

### Adding or updating brand content

Edit `data/brands.json` and re-run `python ingest.py` to refresh the vector store. Each brand supports multiple named versions (`baseline`, `optimized`) for comparing content strategies.

## Project structure

```
src/
  config.py                  models, categories, prompt templates
  engines/
    query_engine.py          OpenRouter client, Condition A
    rag_engine.py            RAG-augmented engine, Conditions B and C
  analyzers/
    brand_extractor.py       mention detection, position ranking
    sentiment_analyzer.py    LLM-as-judge sentiment evaluation
  rag/
    ingestor.py              embed brand docs into ChromaDB
    retriever.py             semantic top-k retrieval by category and version
  utils/
    results_reporter.py      DataFrame conversion, console tables, JSON export
data/
  brands.json                brand content for 9 brands across 2 categories
app.py                       Streamlit interactive dashboard
run_baseline.py              Condition A experiment runner
run_rag.py                   Conditions B and C experiment runner
run_pseudo.py                NexaCRM pseudo-brand experiment
run_feedback_loop.py         closed-loop LLM content optimizer
run_embedding_analysis.py    t-SNE and cosine distance analysis
ingest.py                    vector store population script
```

## Brands and categories

9 brands across two categories:

- **CRM Software:** HubSpot, Salesforce, Copper, Less Annoying CRM, Freshsales
- **Project Management:** Asana, Jira, Notion, ClickUp

## Key dependencies

- **openai** - OpenRouter uses an OpenAI-compatible API
- **chromadb** - vector store for RAG retrieval
- **sentence-transformers** - document and query embedding (all-MiniLM-L6-v2)
- **streamlit** - interactive dashboard
- **plotly** - dashboard visualizations
- **pandas** - tabular analysis
- **scikit-learn** - metrics and clustering
- **spacy** - NLP utilities
- **matplotlib** - publication figure generation

## Research context

This project investigates three questions:

1. Do the content optimization strategies from the KDD 2024 GEO paper (quotation addition, statistics injection, source citations) generalize across current-generation LLMs?
2. Does RAG-augmented context injection improve brand visibility more than the model's parametric knowledge alone?
3. How does query specificity (vague vs. detailed prompts) influence which brands surface?

## License

MIT
