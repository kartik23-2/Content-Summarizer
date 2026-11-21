# AI Tool Disclosure Statement

In compliance with the Generative AI Intern Assessment guidelines, this document transparently outlines the AI coding assistants used during the development of this project, their specific roles, and the engineering oversight applied.

---

## 1. AI Tools & Models Utilized

- **Primary AI Assistant**: **Antigravity AI** (Google DeepMind), powered by **Gemini 2.5/3.8 Flash** models.
- **Development Environment**: Antigravity IDE on Windows.

---

## 2. Scope of AI Assistance

AI coding assistance was utilized as a productivity accelerator across the following specific project areas:

### 2.1 Scaffolding and Boilerplate Implementation
- Generating initial function signatures, class scaffolding, and boilerplate imports for `VectorStore`, `Retriever`, and Streamlit UI components.
- Formulating regex patterns for text normalization, carriage return stripping, and coreference detection in conversational queries.

### 2.2 Test Suite & Synthetic Document Generation
- Generating synthetic benchmark texts in `sample_docs/` (`doc_a_os_scheduling.txt`, `doc_b_os_realtime.txt`, and `doc_c_process_memory.txt`) with realistic technical operating systems content, intentional Round Robin time-quantum discrepancies (10ms vs. 20ms), and benchmark Table 3.1.
- Writing unit test cases in `tests/test_rag.py` to achieve comprehensive code coverage for chunking, normalization, threshold gating, and citation validation.

### 2.3 Documentation & Visual Artifacts
- Generating initial Markdown templates and Mermaid sequence and flowchart diagrams for `implementation.md`, `Readme.md`, and `technical_decisions.md`.
- Formatting benchmark evaluation tables and drafting demonstration video timing cues.

---

## 3. Human Engineering Oversight, Decision-Making & Verification

All architectural design choices, mathematical models, empirical calibrations, and code integrations were rigorously guided, reviewed, and validated by the candidate:

1. **Dual-Threshold Relevance Gating Architecture**:
   - The candidate recognized that standard $L_2$ Euclidean distance was uncalibrated and that an unmeasured 0.65 threshold on `all-MiniLM-L6-v2` would cause severe false abstentions on short queries. The candidate designed the unit L2-normalization approach on FAISS `IndexFlatIP` and established the empirical 0.30 hard floor / 0.50 soft zone framework.
2. **Programmatic Chunk-ID Citation Validation**:
   - Rather than relying on fragile regex checks over free-text model outputs, the candidate designed the programmatic validation engine: requiring the LLM to output structured JSON with specific chunk IDs (`[doc_01_p1_c01]`), which are verified against the retrieved chunk set in code before rendering citations to the user.
3. **4-Tier LLM Call Optimization Strategy**:
   - The candidate conceptualized and enforced the cost-reduction hierarchy (On-Demand Summarization, Document-Scoped Semantic Caching, Heuristic Follow-up Filtering, and Relevance Early-Exit), successfully reducing LLM invocations by over 30%.
4. **Verification & Code Quality**:
   - Every module was executed, verified, and debugged through automated pytest suites (9/9 passed) and end-to-end benchmark runs (10/10 queries evaluated).

---

## 4. Assessment Readiness & Code Ownership

The candidate has full ownership, comprehensive understanding, and independent command of every architectural decision, data structure, and line of code in this repository, and is prepared to defend all design choices, trade-offs, and empirical measurements during the technical assessment interview.
