"""
Smoke test for the RAG pipeline.

Validates end-to-end: ingest → retrieve → RAG query → brand extraction.
Uses a temporary ChromaDB path so it does not touch the real vector store.

Run with:
    python test_rag.py
"""

import tempfile

from src.config import OPENROUTER_API_KEY
from src.rag.ingestor import ingest
from src.rag.retriever import retrieve
from src.engines.rag_engine import query_model_rag
from src.analyzers.brand_extractor import extract_brands

TRACKED = {
    "established": ["Salesforce", "HubSpot", "Zoho CRM", "Microsoft Dynamics 365"],
    "emerging": ["Copper", "Close", "Freshsales", "Less Annoying CRM", "Pipedrive"],
}

QUERY = "What are the best CRM tools for a small startup?"
CATEGORY = "crm_software"


def main():
    assert OPENROUTER_API_KEY, "Missing OPENROUTER_API_KEY in .env"
    print(f"API key: {OPENROUTER_API_KEY[:15]}...\n")

    # 1 - ingest into a temp db so the smoke test is self-contained
    with tempfile.TemporaryDirectory() as tmp_db:
        print("Step 1: Ingesting brands.json into temporary ChromaDB ...")
        count = ingest(brands_file="data/brands.json", db_path=tmp_db)
        assert count > 0, "Ingest returned 0 documents"
        print(f"  Ingested {count} documents\n")

        # 2 - retrieval
        print(f"Step 2: Retrieving top-3 docs for query: '{QUERY}' ...")
        hits = retrieve(QUERY, category=CATEGORY, top_k=3, db_path=tmp_db)
        assert len(hits) > 0, "Retrieval returned no results"
        for hit in hits:
            print(f"  [{hit['score']:.4f}] {hit['brand']}: {hit['content'][:80]}...")
        print()

        # 3 - RAG query (single model, cheap)
        print("Step 3: Querying model with RAG-augmented prompt ...")
        result = query_model_rag(
            QUERY,
            model_key="gpt-5.4-mini",
            model_id="openai/gpt-5.4-mini",
            category=CATEGORY,
            top_k=3,
            db_path=tmp_db,
        )
        assert result["status"] == "success", f"Query failed: {result.get('error')}"
        assert result["condition"] == "rag"
        assert len(result["rag_hits"]) > 0, "No RAG hits recorded in result"
        print(f"  Response ({len(result['response'])} chars):")
        print(f"  {result['response'][:300]}...\n")
        print(f"  RAG hits injected: {[h['brand'] for h in result['rag_hits']]}\n")

        # 4 - brand extraction
        print("Step 4: Extracting brand mentions ...")
        extraction = extract_brands(result["response"], TRACKED)
        print(f"  Brands detected: {extraction['brands_mentioned']}")
        print(f"  Positions:       {extraction['brand_positions']}")
        print(f"  By tier:         {extraction['mentions_by_tier']}\n")

    print("All checks passed.")


if __name__ == "__main__":
    main()
