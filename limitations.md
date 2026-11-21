# Known Limitations and Proposed Improvements

This document provides a rigorous engineering analysis of the known limitations of the current Grounded Document Q&A System, along with actionable improvements for production scaling.

---

## 1. Architectural & Algorithmic Limitations

### 1.1 Bi-Encoder Negation Blindness
- **Description**: Dense bi-encoder models such as `all-MiniLM-L6-v2` utilize average pooling over transformer token representations. In dense vector space, polar opposites differing only by a negation particle (e.g., *"Preemption is permitted"* vs. *"Preemption is not permitted"*) often exhibit high cosine similarity ($S_c > 0.90$).
- **Impact on RAG**: If a user asks *"Is preemption prohibited under Condition 1?"*, vector search may retrieve chunks stating *"Preemption is permitted"*.
- **Current Safeguard**: The system relies on the LLM's closed-domain prompt to inspect the retrieved text and abstain if the exact statement is unsupported.
- **Proposed Improvement**: Integrate a **Cross-Encoder Re-ranker** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) over top-10 dense candidates. Cross-encoders process query and passage jointly with cross-attention, accurately distinguishing negative polarity.

### 1.2 Lack of Sparse Lexical Search (Hybrid Retrieval)
- **Description**: The current retrieval engine relies exclusively on dense semantic vector embeddings.
- **Impact on RAG**: While dense embeddings excel at paraphrases and conceptual similarity, they struggle with exact alphanumeric identifiers, rare function names (`pthread_mutex_lock`), error codes (`0x80070005`), or acronyms not well-represented in general pre-training corpora.
- **Proposed Improvement**: Implement **Hybrid Search with Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{Dense}, \text{BM25}\}} \frac{1}{k + \text{Rank}_m(d)}$$
  Combining BM25 with FAISS ensures both exact keyword precision and semantic recall.

### 1.3 In-Memory Semantic Cache Lifetime
- **Description**: The semantic query cache is stored in-memory (`st.session_state`).
- **Impact**: Cache entries are scoped to the active document collection hash, but restarting the Streamlit server or refreshing the session flushes the cache.
- **Proposed Improvement**: Persist the semantic cache into an embedded key-value store (e.g., **SQLite** or **DuckDB**) or external **Redis** instance with TTL (Time-To-Live) eviction.

---

## 2. Ingestion & Preprocessing Limitations

### 2.1 OCR and Scanned PDF Ingestion
- **Description**: Image-based or scanned PDFs without embedded text layers produce zero extractable text via `PyPDF2`.
- **Current Safeguard**: The preprocessor measures page character density ($<50$ characters) and issues a clean warning alert, skipping empty vectors to prevent index pollution.
- **Proposed Improvement**: Bundle a local offline OCR engine (e.g., **Tesseract** or **PaddleOCR**) or utilize vision-language document models (such as **ColPali** or **Nougat**) to extract text, tables, and diagrams directly from page bitmaps without third-party API dependencies.

### 2.2 Complex Multi-Column & Table Layouts
- **Description**: Standard PDF text extraction reads text streams linearly, which can interleave columns in multi-column research papers or flatten complex tables into disorganized strings.
- **Current Safeguard**: Recursive chunking uses double newlines to isolate blocks, and Table 3.1 was parsed as structured lines.
- **Proposed Improvement**: Integrate layout-aware document parsers such as **pdfplumber**, **Docling**, or **Marker** to preserve markdown table structures and column reading orders.

---

## 3. Conversational Memory & Scale Limitations

### 3.1 Sliding-Window Memory Truncation
- **Description**: The conversation manager maintains a sliding window of the last 6 turns (3 query-response pairs) to avoid context window overflow during query reformulation.
- **Impact**: Deep conversational references to statements made 10 turns earlier are dropped from reformulation context.
- **Proposed Improvement**: Implement a hierarchical memory model that periodically summarizes older conversational turns into a condensed episodic memory state.

### 3.2 Single-Node Vector Storage Scalability
- **Description**: The current vector store uses FAISS `IndexFlatIP`, performing exact brute-force search over in-memory vectors.
- **Impact**: While search latency is $<5\text{ms}$ for collections up to 50,000 chunks, memory consumption and query latency scale linearly ($O(N)$).
- **Proposed Improvement**: For collections exceeding 100,000 documents, migrate to an approximate nearest neighbor (ANN) index such as **FAISS `IndexHNSWFlat`** or a distributed vector database like **ChromaDB**, **Qdrant**, or **Milvus**.

---

## 4. Summary Matrix of Limitations & Roadmaps

| Area | Known Limitation | Current Mitigation | Production Upgrade Path |
| :--- | :--- | :--- | :--- |
| **Embedding** | Negation blindness in bi-encoder. | Closed-domain LLM prompt verification. | Cross-Encoder re-ranker (`ms-marco-MiniLM`). |
| **Search** | Weak on rare alphanumeric codes. | 800-char semantic context overlap. | Hybrid BM25 + Dense FAISS via RRF. |
| **OCR** | Image-only PDFs flagged with warning. | Graceful user warning; no crash. | Local Tesseract / PaddleOCR fallback. |
| **Cache** | Session-bound in-memory cache. | Collection hash scoping. | SQLite / Redis persistent semantic cache. |
| **Memory** | 6-turn sliding window truncation. | Heuristic short-query reformulation. | Summarized episodic conversational memory. |
| **Index** | Flat brute-force search ($O(N)$). | Inverted metadata array + disk persistence. | HNSW graph index / distributed vector DB. |
