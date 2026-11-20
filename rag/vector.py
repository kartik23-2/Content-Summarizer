import os
import json
import faiss
import numpy as np


class VectorStore:
    """
    FAISS-backed vector database using IndexFlatIP on L2-normalized vectors
    (exact Cosine Similarity) with linked metadata, near-duplicate suppression,
    and disk persistence.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []

    def count(self) -> int:
        return self.index.ntotal

    def is_empty(self) -> bool:
        return self.index.ntotal == 0

    def add(self, embeddings: np.ndarray, chunks: list[dict], duplicate_threshold: float = 0.97) -> int:
        """
        Add chunks and their normalized embeddings to the index.
        Suppresses near-duplicate chunks (cosine similarity > duplicate_threshold).
        Returns number of added chunks.
        """
        if embeddings is None or len(embeddings) == 0:
            return 0

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        added_count = 0

        for i, emb in enumerate(embeddings):
            chunk = chunks[i]

            # Near-duplicate suppression if index is non-empty
            if self.index.ntotal > 0 and duplicate_threshold < 1.0:
                dists, _ = self.index.search(emb.reshape(1, -1), 1)
                top_sim = float(dists[0][0])
                if top_sim >= duplicate_threshold:
                    # Near duplicate found - skip adding duplicate chunk vector
                    continue

            self.index.add(emb.reshape(1, -1))
            self.metadata.append(chunk)
            added_count += 1

        return added_count

    def search(self, query_embedding: np.ndarray, k: int = 4) -> tuple[list[float], list[dict]]:
        """
        Search for top-k nearest chunks using inner product on normalized embeddings.
        Returns: (scores: list[float], chunks: list[dict])
        """
        if self.is_empty():
            return [], []

        query_emb = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)

        search_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_emb, search_k)

        out_scores = []
        out_chunks = []

        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                out_scores.append(float(score))
                out_chunks.append(self.metadata[idx])

        return out_scores, out_chunks

    def save(self, directory_path: str):
        """Persist FAISS index and metadata to disk."""
        os.makedirs(directory_path, exist_ok=True)
        index_path = os.path.join(directory_path, "faiss.index")
        meta_path = os.path.join(directory_path, "metadata.json")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(cls, directory_path: str) -> "VectorStore":
        """Load persisted FAISS index and metadata from disk."""
        index_path = os.path.join(directory_path, "faiss.index")
        meta_path = os.path.join(directory_path, "metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Index or metadata not found in '{directory_path}'")

        loaded_index = faiss.read_index(index_path)
        store = cls(dim=loaded_index.d)
        store.index = loaded_index

        with open(meta_path, "r", encoding="utf-8") as f:
            store.metadata = json.load(f)

        return store
