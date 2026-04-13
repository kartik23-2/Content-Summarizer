from rag.embedder import get_embeddings

def retrieve(query, vector_store, k=3):
    query_embedding = get_embeddings([query])[0]
    return vector_store.search(query_embedding, k=k)