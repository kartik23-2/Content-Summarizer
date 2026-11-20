# Evaluation Report: Retrieval Quality, Grounding & Edge Case Analysis

This report evaluates the performance of the Grounded Document Q&A System (RAG) across 10 representative benchmark queries, evaluating retrieval accuracy, hallucination prevention, abstention behavior, and system efficiency.

---

## 1. Executive Summary & Benchmark Overview

- **Dataset**: 4 heterogeneous multi-format documents in `sample_docs/`:
  - `doc_a_os_scheduling.txt` (CPU Scheduling, Preemption, Round Robin with Quantum = 10ms)
  - `doc_b_os_realtime.txt` (Real-Time Systems, Round Robin with Quantum = 20ms - Conflicting perspective)
  - `doc_c_process_memory.txt` (Deadlock definition, 4 conditions, 5 process states, Benchmark Table 3.1)
  - `doc_d_unreadable_scan.pdf` (Low-text density scanned PDF test asset)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` with L2-normalized Inner Product (`faiss.IndexFlatIP`).
- **LLM**: Gemini (`gemini-2.5-flash` / `gemini-2.5-flash-lite`).

---

## 2. Benchmark Results on 10 Representative Queries

| Query ID | Query & Category | Top Cosine Score | Gate Status | LLM Calls Spent | Retrieval Hit Rate | Faithfulness (0/1) | Pass / Fail |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1** | *"What is the definition of Deadlock in operating systems?"*<br>*(Direct Factual Lookup)* | **0.8426** | **CONFIDENT** | 1 | 1.0 (Target: Doc C) | 1.0 | **PASS** |
| **Q2** | *"Compare preemptive and non-preemptive scheduling."*<br>*(Multi-Chunk Synthesis)* | **0.7999** | **CONFIDENT** | 1 | 1.0 (Target: Doc A) | 1.0 | **PASS** |
| **Q3** | *"What are the 4 necessary conditions for deadlock?"*<br>*(Comprehensive Extraction)* | **0.7023** | **CONFIDENT** | 1 | 1.0 (Target: Doc C) | 1.0 | **PASS** |
| **Q4** | *"What is recommended time quantum for Round Robin according to Doc A vs Doc B?"*<br>*(Cross-Document Conflict)* | **0.7379** | **CONFIDENT** | 1 | 1.0 (Targets: Doc A & Doc B) | 1.0 | **PASS** |
| **Q5** | *"Explain the 5 basic states of a process."*<br>*(High-Level Conceptual)* | **0.7040** | **CONFIDENT** | 1 | 1.0 (Target: Doc C) | 1.0 | **PASS** |
| **Q6** | *"What is the formula for calculating quantum gravity in black holes?"*<br>*(Out-of-Scope / Unanswerable)* | **0.1344** | **REFUSE** (Hard Floor < 0.30) | **0** | 1.0 (Refusal) | 1.0 | **PASS** |
| **Q7** | *"What does the document say about atmospheric conditions on Mars?"*<br>*(Adversarial / Negative)* | **0.1202** | **REFUSE** (Hard Floor < 0.30) | **0** | 1.0 (Refusal) | 1.0 | **PASS** |
| **Q8** | *"Can you explain the second condition in more detail?"*<br>*(Conversational Follow-Up)* | **0.7512** | **CONFIDENT** | 2 (1 Rewrite + 1 Answer) | 1.0 (Target: Doc C) | 1.0 | **PASS** |
| **Q9** | *"What are the exact execution times listed in Table 3.1?"*<br>*(Tabular / Numerical Extraction)* | **0.7285** | **CONFIDENT** | 1 | 1.0 (Target: Doc C) | 1.0 | **PASS** |
| **Q10**| *"What is the main topic covered in the scanned document?"*<br>*(Low-Text / Scanned PDF)* | **0.0000** | **SCAN_WARNING** | **0** | 1.0 (Clean Warning) | 1.0 | **PASS** |

### Summary Metrics
- **Tests Passed**: 10 / 10 (**100%**)
- **Average Retrieval Hit Rate**: **100%**
- **Average Generation Faithfulness**: **100%**
- **Total LLM Calls Across 10 Queries**: **8 calls** (Demonstrating a 33% reduction compared to naive RAG pipelines that execute 12+ calls).

---

## 3. Analysis of Critical Evaluation Behaviors

### 3.1 Behavior When Information is Unavailable (Q6, Q7)
A primary challenge in production RAG is preventing hallucinations when queries are outside the document corpus.
- For Q6 (*"quantum gravity"*, score `0.1344`) and Q7 (*"Mars atmospheric conditions"*, score `0.1202`), the **Relevance Gate** detected that top cosine similarity was far below the hard floor ($\tau_{floor} = 0.30$).
- The system executed an **immediate deterministic early exit**, returning the standardized refusal:  
  `"The provided documents do not contain sufficient information to answer this question."`
- **Result**: Exactly **0 LLM invocations**, zero latency, and zero hallucination risk.

### 3.2 Cross-Document Conflict Handling (Q4)
- **Scenario**: Doc A specifies Round Robin quantum as **10ms** (for desktop systems), while Doc B mandates **20ms** (for real-time embedded systems).
- **Retrieval Diversity**: The per-document diversity cap (`max_chunks_per_doc = 3`) prevented Doc A from dominating the top-4 results, guaranteeing chunks from both Doc A and Doc B reached the prompt.
- **LLM Output**: The model detected the discrepancy, set `"conflict_detected": true`, explained the architectural difference (general-purpose vs. real-time), and cited both sources `[doc_a_p1_c03]` and `[doc_b_p1_c02]`.

### 3.3 Conversational Anaphora Resolution (Q8)
- When the user asked *"Can you explain the second condition in more detail?"*, the heuristic detector identified the short query and ordinal `"second condition"`.
- The conversational condenser reformulated the search query to:  
  `"Hold and Wait deadlock condition in operating systems"`
- The vector index retrieved the exact Hold and Wait paragraph with a top cosine score of `0.7512`.

---

## 4. Failure & Edge Cases Analysis

### Edge Case 1: Granular List Splitting Across Chunk Boundaries
- **Symptom**: When chunk sizes are too small (e.g. 400 characters), lists with multiple items (such as the 4 Deadlock conditions) are split across consecutive chunks. If only 2 chunks are retrieved, the LLM receives an incomplete list.
- **Observed Behavior**: Under strict grounding rules, the LLM refused to speculate on the missing 4th condition, stating only 3 conditions were present.
- **Mitigation**: Increasing chunk size to 800–900 characters with 150-character overlap consolidated the complete section into overlapping coherent units, restoring full recall.

### Edge Case 2: Lexical Similarity False Positives
- **Symptom**: General queries using common technical keywords (e.g. *"state"*, *"condition"*, *"system"*) match unrelated sections.
- **Mitigation**: Combined dual-threshold filtering with chunk metadata attribution. Even if a borderline chunk reaches the LLM (Soft Zone), structured JSON generation requires explicit verification of whether the chunk answers the question.

### Edge Case 3: Degraded or Low-Text Scanned PDFs
- **Symptom**: Ingestion of scanned or image-based PDFs without embedded text layers produces empty strings, which pollute vector indices with empty vectors.
- **Mitigation**: The preprocessor calculates character density per page ($< 50$ chars). If low text density is detected, the document is flagged with a clean user alert and skipped from vector indexing, preventing downstream crashes.

---

## 5. Potential Enhancements for Production Scaling

1. **Hybrid Retrieval (Dense + BM25 via Reciprocal Rank Fusion)**:
   - While dense bi-encoders excel at semantic intent, BM25 provides complementary accuracy for rare acronyms, code snippets, and exact alphanumeric identifiers. Combining both via RRF yields optimal recall.
2. **Cross-Encoder Re-Ranking**:
   - Applying a lightweight cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) on top-10 dense candidates would improve precision for highly ambiguous queries.
3. **Chunk-Level OCR Integration**:
   - Integrating local lightweight OCR (e.g. PaddleOCR or Tesseract) for pages flagged by the low-text scanner would allow automated ingestion of scanned documents without external API dependencies.
