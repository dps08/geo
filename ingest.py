"""
One-time script to embed brand content and populate the ChromaDB vector store.

Run this before any RAG experiment:
    python ingest.py

Re-running is safe: documents are upserted (overwritten, not duplicated).
"""

import json
import logging
from pathlib import Path

from src.rag.ingestor import ingest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

BRANDS_FILE = "data/brands.json"
DB_PATH = "data/chroma_db"


def main():
    log.info("Starting ingestion from %s ...", BRANDS_FILE)

    data = json.loads(Path(BRANDS_FILE).read_text())

    # Print a summary of what will be ingested before embedding.
    print("\nBrands to ingest:")
    total = 0
    for category, brands in data.items():
        print(f"\n  [{category}]")
        for brand, brand_data in brands.items():
            versions = list(brand_data["versions"].keys())
            print(f"    {brand:<30} versions: {versions}")
            total += len(versions)

    print(f"\nTotal documents: {total}\n")

    count = ingest(brands_file=BRANDS_FILE, db_path=DB_PATH)

    print(f"\nIngestion complete. {count} document(s) stored in '{DB_PATH}'.")
    print("Run 'python run_rag.py --quick --skip-sentiment' to test the pipeline.")


if __name__ == "__main__":
    main()
