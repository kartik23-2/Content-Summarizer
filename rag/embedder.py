import numpy as np
from sentence_transformers import SentenceTransformer

_model_instance = None

def get_embedder_model():
    """Lazy loader for SentenceTransformer model."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_instance


def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length so inner product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Generate L2-normalized embeddings for a list of texts.
    Returns: 2D numpy array of shape (len(texts), 384) with unit norm.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_embedder_model()
    raw_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    if len(raw_embeddings.shape) == 1:
        raw_embeddings = raw_embeddings.reshape(1, -1)
        
    return normalize_l2(raw_embeddings.astype(np.float32))
