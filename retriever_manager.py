from functools import lru_cache
from typing import Any

from algorithms.bm25_retriever import BM25Retriever
from algorithms.cosine_retriever import CosineRetriever
from algorithms.euclidean_retriever import EuclideanRetriever
from algorithms.hnsw_retriever import HNSWRetriever
from algorithms.hybrid_retriever import HybridRetriever
from utils.data_loader import load_qa_dataset, prepare_corpus

ALGORITHM_LABELS = {
    "cosine": "Cosine Similarity",
    "euclidean": "Euclidean Distance",
    "bm25": "BM25",
    "hnsw": "HNSW (Approximate NN)",
    "hybrid": "Hybrid (BM25 + Cosine)",
}


@lru_cache(maxsize=1)
def get_retrievers() -> dict[str, Any]:
    """Load dataset once and fit all retrievers."""
    records = load_qa_dataset()
    documents, _, metadata = prepare_corpus(records)

    retrievers = {
        "cosine": CosineRetriever(),
        "euclidean": EuclideanRetriever(),
        "bm25": BM25Retriever(),
        "hnsw": HNSWRetriever(),
        "hybrid": HybridRetriever(alpha=0.5),
    }

    for retriever in retrievers.values():
        retriever.fit(documents, metadata)

    return retrievers


def search_all(query: str, top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
    """Run query against every algorithm."""
    retrievers = get_retrievers()
    return {name: retriever.search(query, top_k=top_k) for name, retriever in retrievers.items()}


def search_single(algorithm: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Run query against one algorithm."""
    retrievers = get_retrievers()
    if algorithm not in retrievers:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return retrievers[algorithm].search(query, top_k=top_k)
