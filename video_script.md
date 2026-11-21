# 🎬 Demonstration Video Script: Grounded Document Q&A Assistant (RAG)

**Target Duration**: 4:30 – 5:00 minutes  
**Target Audience**: Technical Assessment Evaluators / Generative AI Engineering Team  
**Presenter**: Candidate (Generative AI Intern Applicant)  
**Demo Application**: `streamlit run app.py`  

---

## 📋 Pre-Recording Setup Checklist
- [ ] Terminal 1: Run `streamlit run app.py` (App loaded at `http://localhost:8501`).
- [ ] Browser window: Fullscreen on the Streamlit UI with clear zoom (100–110%).
- [ ] File explorer or folder open with `sample_docs/`:
  - `doc_a_os_scheduling.txt`
  - `doc_b_os_realtime.txt`
  - `doc_c_process_memory.txt`
  - `doc_d_unreadable_scan.pdf`
- [ ] Microphone tested and terminal with `pytest` ready in background.

---

## ⏱️ Video Script Timeline & Flow

```
+-----------------------------------------------------------------------------------+
| 0:00 - 0:45 | Scene 1: Introduction, Problem Statement & Architecture             |
| 0:45 - 1:30 | Scene 2: Multi-Document Ingestion & On-Demand Summarization          |
| 1:30 - 2:15 | Scene 3: Factual Grounding & Programmatic Chunk-ID Citations        |
| 2:15 - 2:50 | Scene 4: Multi-Turn Conversational Memory & Coreference Resolution  |
| 2:50 - 3:30 | Scene 5: Cross-Document Conflict Handling (Doc A vs Doc B)          |
| 3:30 - 4:10 | Scene 6: Zero-LLM Call Optimizations (Dual-Threshold Gate & Cache)  |
| 4:10 - 4:45 | Scene 7: Edge Cases (Scanned PDF) & Automated Test Evaluation       |
| 4:45 - 5:00 | Scene 8: Conclusion & Engineering Summary                           |
+-----------------------------------------------------------------------------------+
```

---

### Scene 1: Introduction & Architecture Overview (0:00 – 0:45)

**Visual**:
- Camera on presenter or shared screen showing the Streamlit landing page and architecture diagram from `Readme.md`.

**Voiceover**:
> *"Hello! Welcome to the technical demonstration of my Grounded Document Question-Answering System built for the Generative AI assessment.*
>
> *The goal of this project isn't simply connecting an LLM to a vector database. It is designing a resilient, hallucination-proof retrieval pipeline that understands collections of heterogeneous documents, resolves multi-turn conversational context, detects cross-document conflicts, and strictly refuses unanswerable queries.*
>
> *Architecturally, the pipeline uses SentenceTransformers with L2-normalized FAISS IndexFlatIP for exact cosine similarity, an empirically calibrated dual-threshold relevance gate, programmatic chunk-ID citation validation, and a multi-tier optimization framework that eliminates unnecessary LLM calls. Let's see it in action."*

---

### Scene 2: Multi-Document Ingestion & On-Demand Summarization (0:45 – 1:30)

**On-Screen Action**:
1. Drag and drop three documents from `sample_docs/` into the sidebar uploader:
   - `doc_a_os_scheduling.txt`
   - `doc_b_os_realtime.txt`
   - `doc_c_process_memory.txt`
2. Show the real-time processing indicator: `Indexed 3 documents (14 distinct chunks)`.
3. Point to the sidebar breakdown showing pages and chunks per file.
4. Click the **"Generate Document Summary"** button.

**Voiceover**:
> *"In the sidebar, I am uploading three distinct documents covering operating systems principles, real-time control systems, and process concurrency.*
>
> *The ingestion engine performs character offset tracking, Unicode cleaning, and recursive semantic chunking at 800 characters with 150-character overlap. Each chunk embedding is unit-normalized and indexed in FAISS with near-duplicate suppression.*
>
> *Notice our first LLM call optimization: unlike naive RAG systems that eagerly burn an expensive full-document LLM call on every upload, summarization is strictly on-demand. When I click 'Generate Document Summary', the system generates a structured overview citing specific document themes with exactly one LLM call."*

---

### Scene 3: Factual Grounding & Programmatic Citations (1:30 – 2:15)

**On-Screen Action**:
1. In the chat input, type:  
   `What is the definition of Deadlock in operating systems?`
2. Hit Enter.
3. Show the response streaming/rendering.
4. Expand the **"🔍 Verified Source References & Citations"** accordion.
5. Highlight the metrics badge: `Top Similarity: 0.843 | Status: CONFIDENT | LLM Invocations: 1`.

**Voiceover**:
> *"Now let's ask a direct factual question: 'What is the definition of Deadlock in operating systems?'*
>
> *Our retriever embeds the query, queries FAISS, and yields a top cosine similarity score of 0.843—firmly inside our Confident Zone (above 0.50).*
>
> *Notice the grounding: the response directly cites `[doc_c_process_memory_txt_p1_c01]`. Rather than relying on fuzzy regex, our backend validates that this chunk ID existed in the retrieved set before rendering it. In the expandable citation drawer below, you can inspect the exact source file, page number, and text snippet that substantiated this claim."*

---

### Scene 4: Conversational Memory & Coreference Reformulation (2:15 – 2:50)

**On-Screen Action**:
1. Type a follow-up query:  
   `What are the 4 necessary conditions for deadlock?`
2. Wait for response listing Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait.
3. Immediately ask an anaphoric follow-up:  
   `Can you explain the second condition in more detail?`
4. Show the response elaborating on *Hold and Wait*.
5. Highlight the metrics showing conversational reformulation.

**Voiceover**:
> *"Now let's test multi-turn conversational reasoning. First, I ask for the 4 necessary conditions for deadlock. The system extracts all four from Document C.*
>
> *Next, I ask an ambiguous follow-up: 'Can you explain the second condition in more detail?'*
>
> *Notice that this query contains no keywords—just the pronoun 'it' and ordinal 'second condition'. A naive search would fail completely. Our conversational manager detects that this query requires context, reformulates it to 'Hold and Wait deadlock condition in operating systems', and retrieves the exact paragraph from Document C without breaking conversational flow."*

---

### Scene 5: Cross-Document Conflict Handling (2:50 – 3:30)

**On-Screen Action**:
1. Type the cross-document query:  
   `What is the recommended time quantum for Round Robin scheduling according to Doc A vs Doc B?`
2. Show the response highlighting both documents:
   - Document A: 10ms for general desktop OS.
   - Document B: 20ms for real-time industrial control.
3. Point out the warning banner: `⚠️ Cross-Document Discrepancy Detected`.

**Voiceover**:
> *"A key requirement of the assessment is handling documents with duplicate or conflicting information.*
>
> *In `sample_docs`, Document A recommends a 10ms Round Robin quantum for desktop operating systems, while Document B mandates a 20ms quantum for real-time systems.*
>
> *To prevent one document from crowding out another, our retriever enforces a per-document diversity cap. The LLM prompt specifically checks for conflicting claims across sources. As you can see, the system detected the conflict, flagged it with a discrepancy notice, and explained both architectural rationales while citing both documents."*

---

### Scene 6: Zero-LLM Call Optimizations (Dual-Threshold Gate & Cache) (3:30 – 4:10)

**On-Screen Action**:
1. Type an out-of-scope query:  
   `What is the formula for calculating quantum gravity in black holes?`
2. Show the immediate response:  
   `"The provided documents do not contain sufficient information to answer this question."`
3. Highlight the metrics badge: `Top Similarity: 0.134 | Status: REFUSE | LLM Invocations: 0`.
4. Now, re-enter the earlier question:  
   `What is the definition of Deadlock in operating systems?`
5. Show the instant response with badge: `Status: CACHE_HIT | Cache Hit: True | LLM Invocations: 0`.

**Voiceover**:
> *"Now let's examine our cost and hallucination optimization.*
>
> *I ask: 'What is the formula for calculating quantum gravity in black holes?' This is completely out-of-scope.*
>
> *Instead of passing weak chunks to an LLM and hoping it doesn't hallucinate, our Calibrated Dual-Threshold Gate measures a top cosine similarity of 0.134—well below our hard floor of 0.30. The system triggers a deterministic early exit: zero latency, zero hallucination risk, and exactly ZERO LLM calls spent.*
>
> *Next, if I re-ask our earlier deadlock question, our document-scoped semantic query cache identifies a 100% semantic match within this document collection, returning the verified answer instantly with ZERO LLM calls."*

---

### Scene 7: Edge Cases & Automated Test Suite (4:10 – 4:45)

**On-Screen Action**:
1. Upload `sample_docs/doc_d_unreadable_scan.pdf` in the sidebar.
2. Show the yellow warning:  
   `Warning: 'doc_d_unreadable_scan.pdf' contains very little extractable text... The document appears to be scanned or image-based.`
3. Switch briefly to the terminal and execute:  
   `pytest tests/test_rag.py -v`
4. Show 9 passed unit tests.
5. Display the evaluation summary table from `evaluate.py`.

**Voiceover**:
> *"For edge case handling, what happens if a user uploads a degraded or image-only scanned PDF? When I upload `doc_d_unreadable_scan.pdf`, our preprocessor checks character density per page. Detecting less than 50 characters, it flags a clean user warning rather than polluting the vector index with empty vectors.*
>
> *In the terminal, our automated pytest suite verifies 9 distinct test cases including L2 normalization, duplicate suppression, persistence, and citation filtering.*
>
> *Furthermore, our automated benchmark runner evaluated 10 representative queries, achieving a 100% retrieval hit rate and 100% faithfulness score."*

---

### Scene 8: Conclusion & Wrap-Up (4:45 – 5:00)

**Visual**:
- Fullscreen on Streamlit app with clean chat history and sidebar.

**Voiceover**:
> *"To summarize: this system delivers strict factual grounding, robust multi-turn conversation, cross-document conflict awareness, and an architectural framework that cuts LLM calls by over 30%.*
>
> *All code, test suites, evaluation reports, and technical decision logs are fully documented and available in the repository. Thank you for your time!"*

---

## 🎥 Recording & Delivery Tips
1. **Pacing**: Speak at a steady, confident, professional pace (~130–140 words per minute).
2. **Cursor Movement**: Move the mouse smoothly to guide the reviewer's eyes to badges, citations, and metrics.
3. **Audio Quality**: Ensure low background noise and clear articulation of technical terms (*"bi-encoder"*, *"IndexFlatIP"*, *"coreference resolution"*).
4. **Resolution**: Record in 1080p (1920x1080) at 30 or 60 fps.
