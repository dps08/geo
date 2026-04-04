"""
Reads brand content from data/brands.json, embeds each document using
sentence-transformers, and persists it in a ChromaDB vector store.

Re-running ingest is safe: documents are upserted so existing entries
are overwritten rather than duplicated.
"""

import json
import logging
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

_COLLECTION_NAME = "geo_brands"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _normalize_id(brand: str, category: str, version: str) -> str:
    """Build a stable, filesystem-safe document ID."""
    raw = f"{brand}_{category}_{version}"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", raw)


def ingest(
    brands_file: str = "data/brands.json",
    db_path: str = "data/chroma_db",
) -> int:
    """
    Embed and store all brand documents from ``brands_file`` in ChromaDB.

    Returns the total number of documents upserted.
    """
    brands_path = Path(brands_file)
    if not brands_path.exists():
        raise FileNotFoundError(f"Brands file not found: {brands_path}")

    data = json.loads(brands_path.read_text())

    log.info("Loading embedding model %s ...", _EMBEDDING_MODEL)
    model = SentenceTransformer(_EMBEDDING_MODEL)

    Path(db_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=_COLLECTION_NAME)

    ids, documents, embeddings, metadatas = [], [], [], []

    for category, brands in data.items():
        for brand, brand_data in brands.items():
            for version, content in brand_data["versions"].items():
                doc_id = _normalize_id(brand, category, version)
                embedding = model.encode(content).tolist()
                ids.append(doc_id)
                documents.append(content)
                embeddings.append(embedding)
                metadatas.append({
                    "brand": brand,
                    "category": category,
                    "version": version,
                })
                log.debug("Prepared %s / %s / %s", category, brand, version)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    log.info("Upserted %d documents into collection '%s'", len(ids), _COLLECTION_NAME)
    return len(ids)
