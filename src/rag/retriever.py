"""
Queries the ChromaDB vector store for brand documents most semantically
relevant to a given prompt, filtered by product category and content version.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

_COLLECTION_NAME = "geo_brands"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Module-level cache so the model and client are not reloaded on every call.
_model: SentenceTransformer | None = None
_clients: dict[str, chromadb.PersistentClient] = {}


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_EMBEDDING_MODEL)
    return _model


def _get_client(db_path: str) -> chromadb.PersistentClient:
    if db_path not in _clients:
        _clients[db_path] = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
    return _clients[db_path]


def retrieve(
    query: str,
    category: str,
    version: str = "baseline",
    top_k: int = 5,
    db_path: str = "data/chroma_db",
) -> list[dict]:
    """
    Return the top-k brand documents most relevant to ``query``.

    Results are filtered to ``category`` and ``version`` so CRM content
    never surfaces in a project-management query and vice versa.

    Each returned dict has keys:
        brand    - brand name
        content  - raw document text
        score    - cosine distance (lower = more similar; 0 is identical)
    """
    if not Path(db_path).exists():
        raise RuntimeError(
            f"ChromaDB not found at '{db_path}'. Run 'python ingest.py' first."
        )

    client = _get_client(db_path)
    collection = client.get_collection(name=_COLLECTION_NAME)

    query_embedding = _get_model().encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"$and": [{"category": category}, {"version": version}]},
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "brand": meta["brand"],
            "content": doc,
            "score": round(dist, 4),
        })

    log.debug(
        "Retrieved %d docs for query=%r category=%s version=%s",
        len(hits), query[:60], category, version,
    )
    return hits
