import os
from rag.embedder import get_embeddings
from rag.vector import VectorStore

# Configurable similarity thresholds
HARD_FLOOR = float(os.getenv("SIMILARITY_HARD_FLOOR", "0.30"))
CONFIDENT_THRESHOLD = float(os.getenv("SIMILARITY_CONFIDENT_THRESHOLD", "0.50"))


class Retriever:
    """
    Retrieval engine with calibrated dual-threshold relevance gating
    and per-document diversity capping (conflict-handling support).
    """

    def __init__(self, hard_floor: float = HARD_FLOOR, confident_threshold: float = CONFIDENT_THRESHOLD):
        self.hard_floor = hard_floor
        self.confident_threshold = confident_threshold

    def retrieve(
        self,
        query: str,
        vector_store: VectorStore,
        k: int = 4,
        max_chunks_per_doc: int = 2
    ) -> dict:
        """
        Retrieve chunks relevant to the query with diversity capping and threshold gating.
        
        Returns dict:
        {
            "query": str,
            "passed_chunks": list[dict],      # Chunks passing hard floor & diversity cap
            "top_score": float,               # Highest cosine similarity score
            "status": "REFUSE" | "SOFT_ZONE" | "CONFIDENT",
            "refusal_reason": str | None
        }
        """
        if vector_store.is_empty():
            return {
                "query": query,
                "passed_chunks": [],
                "top_score": 0.0,
                "status": "REFUSE",
                "refusal_reason": "Vector store is empty. No documents indexed."
            }

        # 1. Embed query (L2 normalized)
        query_emb = get_embeddings([query])[0]

        # 2. Over-fetch to allow diversity capping across multiple documents
        fetch_k = min(k * 3, vector_store.count())
        scores, chunks = vector_store.search(query_emb, k=fetch_k)

        if not scores:
            return {
                "query": query,
                "passed_chunks": [],
                "top_score": 0.0,
                "status": "REFUSE",
                "refusal_reason": "No matching chunks found."
            }

        top_score = scores[0]

        # 3. Check hard floor gate (Early-exit with 0 LLM calls if out-of-scope)
        if top_score < self.hard_floor:
            return {
                "query": query,
                "passed_chunks": [],
                "top_score": top_score,
                "status": "REFUSE",
                "refusal_reason": f"Top similarity ({top_score:.3f}) falls below hard floor ({self.hard_floor:.2f})."
            }

        # 4. Apply per-document diversity cap
        doc_counts = {}
        filtered_chunks = []

        for score, chunk in zip(scores, chunks):
            # Only consider chunks passing hard floor
            if score < self.hard_floor:
                continue

            doc_id = chunk.get("doc_id", "default")
            curr_count = doc_counts.get(doc_id, 0)

            if curr_count < max_chunks_per_doc:
                chunk_copy = dict(chunk)
                chunk_copy["similarity_score"] = round(float(score), 4)
                filtered_chunks.append(chunk_copy)
                doc_counts[doc_id] = curr_count + 1

            if len(filtered_chunks) >= k:
                break

        # 5. Determine status
        status = "CONFIDENT" if top_score >= self.confident_threshold else "SOFT_ZONE"

        return {
            "query": query,
            "passed_chunks": filtered_chunks,
            "top_score": top_score,
            "status": status,
            "refusal_reason": None
        }


def retrieve(query: str, vector_store: VectorStore, k: int = 4) -> list[str]:
    """Legacy helper returning raw text chunks."""
    retriever = Retriever()
    res = retriever.retrieve(query, vector_store, k=k)
    return [c["content"] for c in res["passed_chunks"]]
