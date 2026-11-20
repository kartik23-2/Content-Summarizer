import json

UNIFIED_REFUSAL_MESSAGE = "The provided documents do not contain sufficient information to answer this question."


def build_grounded_qa_prompt(context_chunks: list[dict], question: str) -> str:
    """
    Constructs a closed-domain grounded prompt requiring structured JSON output,
    chunk-ID based citations, and explicit conflict detection.
    """
    formatted_chunks = []
    for c in context_chunks:
        cid = c.get("chunk_id", "unknown_chunk")
        fname = c.get("filename", "unknown_doc")
        page = c.get("page", 1)
        content = c.get("content", "").strip()
        formatted_chunks.append(
            f"=== CHUNK ID: [{cid}] (Source Document: {fname}, Page: {page}) ===\n{content}\n"
        )

    context_str = "\n".join(formatted_chunks)

    prompt = f"""You are a strict, authoritative Document Question-Answering Assistant.
Answer the user's question using ONLY the facts explicitly provided in the Context Chunks below.

CONTEXT CHUNKS:
{context_str}

USER QUESTION:
{question}

INSTRUCTIONS & RULES:
1. Grounding: Rely solely on the provided Context Chunks. Do NOT use outside parametric knowledge or speculate.
2. Structured JSON Output: You must respond ONLY with a single valid JSON object formatted exactly as follows:
{{
    "answerable": true,
    "answer": "Your detailed, grounded answer here. Include inline chunk-ID citations like [{context_chunks[0]['chunk_id'] if context_chunks else 'chunk_id'}].",
    "cited_chunk_ids": ["list", "of", "exact", "chunk_ids", "used"],
    "conflict_detected": false,
    "conflict_explanation": ""
}}

3. Citations: In "cited_chunk_ids", list ONLY the exact Chunk IDs that directly support your statements. Do NOT invent chunk IDs.
4. Unanswerable / Insufficient Information: If the context chunks do NOT contain enough information to answer the question directly, respond with:
{{
    "answerable": false,
    "answer": "{UNIFIED_REFUSAL_MESSAGE}",
    "cited_chunk_ids": [],
    "conflict_detected": false,
    "conflict_explanation": ""
}}

5. Conflict Detection: If multiple documents contain conflicting or contradictory facts (for instance, Document A and Document B specify different numbers, values, or recommendations):
   - Set "conflict_detected": true.
   - Explicitly highlight both conflicting perspectives in your "answer", citing each respective chunk ID.
   - Summarize the discrepancy in "conflict_explanation".

JSON Response:"""
    return prompt


def build_query_reformulation_prompt(history: list[dict], current_query: str) -> str:
    """
    Reformulates follow-up queries or anaphoric references into self-contained search queries.
    """
    history_lines = []
    for turn in history[-4:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = turn.get("content", "").strip()
        history_lines.append(f"{role}: {content}")

    history_str = "\n".join(history_lines)

    return f"""You are an intelligent query reformulation assistant.
Given the previous conversation turns and a follow-up user query, rewrite the follow-up query into a single, self-contained search query suitable for semantic vector retrieval.

RULES:
- Resolve all pronouns ('it', 'they', 'that', 'these', 'the second one', 'the former', 'the latter') and implicit references.
- Preserve the user's original intent.
- Do NOT answer the question. Only output the reformulated query text and nothing else.

CONVERSATION HISTORY:
{history_str}

FOLLOW-UP QUERY:
{current_query}

REFORMULATED QUERY:"""


def build_summary_prompt(chunks: list[dict]) -> str:
    """Generate on-demand structured document summary."""
    sample_text = "\n\n".join([f"[{c.get('filename')} - Page {c.get('page')}]: {c.get('content')}" for c in chunks[:12]])
    return f"""You are an expert academic research assistant.
Provide a clear, structured summary of the key themes, methods, and definitions present in the provided document excerpts.

Excerpts:
{sample_text}

Format your summary with:
- Overview (2-3 sentences)
- Core Concepts & Definitions
- Key Comparative Points / Metrics

Summary:"""


# Legacy compatibility functions
def summary_prompt(text: str) -> str:
    return build_summary_prompt([{"filename": "document", "page": 1, "content": text[:4000]}])


def qa_prompt(context: str, question: str) -> str:
    mock_chunks = [{"chunk_id": "chunk_01", "filename": "document.txt", "page": 1, "content": context}]
    return build_grounded_qa_prompt(mock_chunks, question)
