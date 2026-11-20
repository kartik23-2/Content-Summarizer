import numpy as np
from rag.embedder import get_embeddings


class SemanticCache:
    """
    Document-collection scoped semantic query cache.
    Caches (query_vector, response_data) pairs per collection hash.
    Returns cached response if similarity >= threshold (default 0.95).
    
    Known Limitation:
    Very subtle semantic inversions/negations (e.g. 'Is X allowed?' vs 'Is X not allowed?')
    can share high embedding similarity (>0.94). An exact string comparison check is
    performed alongside cosine similarity.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        # {collection_hash: list of {"query": str, "vector": np.ndarray, "response": dict}}
        self.cache: dict[str, list[dict]] = {}

    def clear(self, collection_hash: str = None):
        """Invalidate cache for a specific collection or all collections."""
        if collection_hash:
            self.cache.pop(collection_hash, None)
        else:
            self.cache.clear()

    def lookup(self, query: str, collection_hash: str) -> dict | None:
        """
        Check if a semantically equivalent query exists in the cache for this collection.
        Returns cached response or None.
        """
        if not collection_hash or collection_hash not in self.cache:
            return None

        entries = self.cache[collection_hash]
        if not entries:
            return None

        clean_query = query.strip().lower()

        # 1. Exact string match check (avoids embedding computation)
        for entry in entries:
            if entry["query"].strip().lower() == clean_query:
                return entry["response"]

        # 2. Semantic vector lookup
        query_emb = get_embeddings([query])[0]
        vectors = np.array([e["vector"] for e in entries], dtype=np.float32)

        # Inner product on normalized vectors = Cosine Similarity
        sims = np.dot(vectors, query_emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.similarity_threshold:
            # Check for negation mismatch between queries
            cached_q = entries[best_idx]["query"].lower()
            q_neg = any(w in clean_query.split() for w in ["not", "no", "never", "unsupported", "cannot"])
            cached_neg = any(w in cached_q.split() for w in ["not", "no", "never", "unsupported", "cannot"])

            if q_neg == cached_neg:
                return entries[best_idx]["response"]

        return None

    def store(self, query: str, collection_hash: str, response: dict):
        """Store query vector and structured response under the collection hash."""
        if not collection_hash:
            return

        if collection_hash not in self.cache:
            self.cache[collection_hash] = []

        query_emb = get_embeddings([query])[0]
        self.cache[collection_hash].append({
            "query": query,
            "vector": query_emb,
            "response": response
        })
