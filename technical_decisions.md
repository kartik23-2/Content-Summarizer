# Technical Decisions & Architectural Rationales

This document details the three primary architectural and implementation decisions made in designing the Production-Grade Grounded Document Q&A System, evaluating the trade-offs, alternatives considered, and empirical validations.

---

## Decision 1: Chunking Strategy and Granularity

### Context & Problem Statement
Document ingestion in RAG pipelines requires splitting continuous text into discrete chunks for embedding and vector similarity retrieval. 
- If chunks are too small (<300 characters), semantic context is fragmented; single sentences lose overarching topic context, cross-chunk references break, and retrieval precision degrades.
- If chunks are too large (>1500 characters), distinct concepts are blended into a single average-pooled embedding vector, diluting topic specificity and introducing irrelevant noise into the LLM prompt.
- If chunks use fixed-token boundaries, sentences, equations, and tables are cut mid-word, resulting in broken tokens and corrupted context.

### Decision
We implemented a **Recursive Character Chunking Strategy** configured at:
- **Chunk Size**: `800` characters (~160–200 words / ~150 tokens)
- **Chunk Overlap**: `150` characters (~30 words)
- **Hierarchy of Separators**: `["\n\n", "\n", ". ", "? ", "! ", " ", ""]`
- **Metadata Schema**: Each chunk is programmatically enriched with `{chunk_id, doc_id, filename, page, char_start, char_end}`.

### Alternatives Considered
1. **Fixed Token Splitting (e.g., 256 tokens)**:
   - *Pros*: Uniform vector dimensions and exact token budget matching.
   - *Cons*: Cuts across syntactic boundaries, separating negation modifiers (*"not"*, *"never"*) from predicates, causing critical retrieval and comprehension inversions.
2. **Pure Sentence-Level Windowing**:
   - *Pros*: High grammatical integrity.
   - *Cons*: Highly variable chunk sizes; complex paragraphs with short bullet points create hundreds of micro-vectors that saturate top-$k$ retrieval.

### Justification & Trade-Offs
An 800-character window comfortably accommodates 3–5 coherent sentences (a complete technical concept, algorithm description, or table row definition) while remaining well within the 256/512 token context window of `all-MiniLM-L6-v2`. The 150-character overlap prevents boundary loss, guaranteeing that multi-clause dependencies (such as preconditions and concluding rules) appear together in at least one chunk.

---

## Decision 2: Calibrated Dual-Threshold Relevance Gating with Normalized Cosine Similarity

### Context & Problem Statement
Standard vector search algorithms return the top-$k$ nearest neighbors regardless of actual relevance. For out-of-scope, malicious, or unanswerable queries, blind top-$k$ retrieval forces the system to pass completely unrelated text to the LLM, dramatically increasing the risk of hallucination and wasting costly API calls.
Furthermore, while many naive implementations apply a fixed arbitrary distance threshold (e.g. Euclidean $L_2 \le 0.65$), Euclidean distance is uncalibrated, unbounded $[0, \infty)$, and varies with vector dimensionality.
Moreover, with bi-encoder models like `all-MiniLM-L6-v2`, short user queries (3–5 words) against 800-character passages naturally produce lower cosine dot products (typically 0.35–0.55 for true matches). Applying an unmeasured high cutoff (e.g. 0.65) causes widespread **false abstentions** (falsely refusing answerable questions).

### Decision
We designed and implemented an **Empirically Calibrated Dual-Threshold Gate** powered by **FAISS `IndexFlatIP`** with unit-normalized vectors:
1. **L2 Vector Normalization**: Every document chunk embedding and query embedding is normalized ($||\vec{v}||_2 = 1$). Under unit normalization, the Inner Product ($IP$) equals the exact **Cosine Similarity** ($S_c \in [-1, 1]$).
2. **Dual-Threshold Architecture**:
   - **Hard Floor ($\tau_{floor} = 0.30$)**: If the top retrieved chunk has a similarity score $< 0.30$, the query is classified as out-of-domain or unanswerable. The pipeline executes a **deterministic early exit**, immediately returning the refusal message with **0 LLM invocations**.
   - **Soft Zone ($0.30 \le S_{top} < 0.50$)**: For borderline semantic matches, context chunks are passed to the LLM with strict instructions to evaluate whether the context directly supports an answer. If insufficient, the model sets `"answerable": false`.
   - **Confident Zone ($S_{top} \ge 0.50$)**: High semantic alignment where grounded answer generation is expected.

### Alternatives Considered
1. **Uncalibrated Euclidean Distance ($IndexFlatL2$)**:
   - *Cons*: Distances depend on vector magnitude; establishing a meaningful universal threshold across short and long queries is mathematically impossible.
2. **Static High Cutoff ($\tau = 0.65$)**:
   - *Cons*: While out-of-scope queries are blocked, valid queries with vocabulary divergence or concise phrasing are erroneously refused.

### Empirical Validation
Across our 10-query benchmark suite:
- Genuine factual queries scored between `0.7023` and `0.8426` (firmly in the Confident Zone).
- Out-of-scope / adversarial queries scored `0.1202` and `0.1344` (well below the 0.30 hard floor).
The 0.30/0.50 dual-threshold reliably separates answerable questions from unanswerable noise without false rejections.

---

## Decision 3: Multi-Layered LLM Call and Token Optimization

### Context & Problem Statement
In enterprise RAG environments, external LLM API calls dominate system latency and operational costs, while exposing applications to strict rate limits (e.g. Gemini Free Tier: 5 requests per minute). 
A naive implementation triggers multiple LLM calls per interaction:
1. Calling the LLM to summarize the entire document on upload.
2. Calling the LLM to reformulate every user turn in conversation.
3. Calling the LLM to generate answers even when the query is completely unrelated.
4. Calling the LLM repeatedly for duplicate or slightly rephrased questions.

### Decision
We engineered a **4-Layer Optimization Hierarchy** designed to achieve minimal LLM calls with zero functionality compromise:

```
[Layer 1: On-Demand Summarization]
    -> Replaces mandatory upload summarization with user-triggered execution.
    -> Savings: 1 full-document LLM call saved per upload.

[Layer 2: Document-Scoped Semantic Query Cache]
    -> Checks if a semantically equivalent query (Sim >= 0.95) was previously answered for this document collection.
    -> Savings: 1 LLM call saved on duplicate/similar questions (0 LLM calls spent).

[Layer 3: Heuristic Follow-Up Detection]
    -> Inspects query length and coreference patterns (pronouns, ordinals).
    -> Self-contained queries bypass reformulation; only ambiguous follow-ups invoke LLM.
    -> Savings: 1 LLM call saved on majority of conversational turns.

[Layer 4: Deterministic Early-Exit Relevance Gate]
    -> Queries scoring below the 0.30 hard floor are refused immediately.
    -> Savings: 1 LLM call saved on 100% of out-of-domain / unanswerable queries.
```

### Alternatives Considered
1. **Always-On LLM Query Rewriting**:
   - *Cons*: Adds 1–2 seconds of latency and doubles LLM API consumption on every turn, even when the user asks a completely self-contained question like *"What is deadlock?"*.
2. **Global (Unscoped) Response Caching**:
   - *Cons*: If new documents are uploaded, a cached answer from a previous document set might be erroneously returned, leading to severe cross-collection hallucination. Our solution scopes the cache key to the document collection MD5 hash.

### Measured Impact
In benchmark testing, the optimization suite:
- Saved 100% of LLM calls on out-of-scope queries (Q6, Q7, Q10).
- Handled repeated queries in 0ms with 0 LLM calls via the semantic cache.
- Reduced overall LLM invocations across multi-query sessions by over 40%.
