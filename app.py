import os
import hashlib
import streamlit as st

from utils.content import load_file_content
from utils.chunks import chunk_document
from rag.embedder import get_embeddings
from rag.vector import VectorStore
from rag.retriever import Retriever
from llm.model import ask_llm, parse_structured_response
from llm.prompt import build_grounded_qa_prompt, build_summary_prompt, UNIFIED_REFUSAL_MESSAGE
from llm.conversation import ConversationManager
from llm.cache import SemanticCache

st.set_page_config(
    page_title="Grounded Document Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize Session State
if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationManager()
if "semantic_cache" not in st.session_state:
    st.session_state.semantic_cache = SemanticCache(similarity_threshold=0.95)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(dim=384)
if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever()
if "collection_hash" not in st.session_state:
    st.session_state.collection_hash = ""
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "on_demand_summary" not in st.session_state:
    st.session_state.on_demand_summary = None

st.title("📚 Intelligent Grounded Document Q&A System")
st.caption("Multi-Document Semantic Retrieval & Hallucination-Grounded Generation (RAG)")

# Sidebar for Document Management
with st.sidebar:
    st.header("📂 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF and TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or multiple documents to build the semantic search index."
    )

    if uploaded_files:
        # Compute combined hash to detect changes
        file_specs = [f"{f.name}_{f.size}" for f in uploaded_files]
        new_hash = hashlib.md5(";;".join(sorted(file_specs)).encode()).hexdigest()

        if new_hash != st.session_state.collection_hash:
            with st.spinner("Processing documents and building vector index..."):
                all_chunks = []
                docs_summary = []
                warnings_list = []

                for uploaded_file in uploaded_files:
                    doc_data = load_file_content(uploaded_file, uploaded_file.name)

                    if doc_data.get("error"):
                        st.error(f"Error in {uploaded_file.name}: {doc_data['error']}")
                        continue

                    if doc_data.get("warning"):
                        warnings_list.append(doc_data["warning"])

                    chunks = chunk_document(doc_data, chunk_size=800, chunk_overlap=150)
                    all_chunks.extend(chunks)
                    docs_summary.append({
                        "name": uploaded_file.name,
                        "pages": len(doc_data["pages"]),
                        "chars": doc_data["total_chars"],
                        "chunks": len(chunks)
                    })

                if all_chunks:
                    chunk_texts = [c["content"] for c in all_chunks]
                    embeddings = get_embeddings(chunk_texts)

                    # Initialize fresh vector store for new collection
                    store = VectorStore(dim=384)
                    added_count = store.add(embeddings, all_chunks, duplicate_threshold=0.97)

                    st.session_state.vector_store = store
                    st.session_state.collection_hash = new_hash
                    st.session_state.processed_docs = docs_summary
                    st.session_state.conversation.clear()
                    st.session_state.chat_messages = []
                    st.session_state.semantic_cache.clear()
                    st.session_state.on_demand_summary = None

                    st.success(f"Indexed {len(docs_summary)} documents ({added_count} distinct chunks)!")
                else:
                    st.error("No valid text chunks could be extracted from uploaded files.")

                for w in warnings_list:
                    st.warning(w)

    # Document Collection Metrics
    if st.session_state.processed_docs:
        st.subheader("Indexed Documents")
        for d in st.session_state.processed_docs:
            st.write(f"📄 **{d['name']}** — {d['pages']} pgs | {d['chunks']} chunks")

        st.divider()
        if st.button("🧹 Clear Chat History"):
            st.session_state.conversation.clear()
            st.session_state.chat_messages = []
            st.rerun()

        # On-Demand Summary Button (Optimized to save LLM calls)
        st.subheader("📑 Document Summarization")
        if st.button("Generate Document Summary"):
            if not st.session_state.vector_store.is_empty():
                with st.spinner("Generating grounded summary (1 LLM call)..."):
                    summary_prompt = build_summary_prompt(st.session_state.vector_store.metadata[:12])
                    summary_res = ask_llm(summary_prompt)
                    st.session_state.on_demand_summary = summary_res

    st.markdown("---")
    st.markdown("**Architecture Highlights:**")
    st.markdown("• Calibrated Dual-Threshold Gating")
    st.markdown("• Programmatic Chunk-ID Citations")
    st.markdown("• Cross-Document Conflict Detection")
    st.markdown("• Document-Scoped Semantic Caching")

# Main Interface: Document Summary (if requested)
if st.session_state.on_demand_summary:
    with st.expander("📑 On-Demand Collection Summary", expanded=True):
        st.markdown(st.session_state.on_demand_summary)

# Main Interface: Chat History
if not st.session_state.processed_docs:
    st.info("👈 Please upload one or more PDF or TXT documents in the sidebar to begin interacting.")
else:
    # Display message history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metrics" in msg:
                m = msg["metrics"]
                st.caption(
                    f"Top Similarity: `{m.get('top_score', 0):.3f}` | Status: `{m.get('status')}` | "
                    f"Cache Hit: `{m.get('cache_hit')}` | LLM Invocations: `{m.get('llm_calls', 0)}`"
                )
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 Verified Source References & Citations"):
                    for s in msg["sources"]:
                        st.markdown(f"**[{s['chunk_id']}]** *{s['filename']}* (Page {s['page']})")
                        st.markdown(f"> {s['snippet']}")

    # User Query Input
    user_query = st.chat_input("Ask a question about the uploaded documents...")

    if user_query:
        # Display user question
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_messages.append({"role": "user", "content": user_query})

        llm_call_count = 0

        # Step 1: Check Conversational Follow-Up / Coreference
        search_query, reformulated = st.session_state.conversation.get_search_query(user_query)
        if reformulated:
            llm_call_count += 1

        # Step 2: Document-Scoped Semantic Cache Lookup
        cached_response = st.session_state.semantic_cache.lookup(search_query, st.session_state.collection_hash)

        if cached_response:
            answer_text = cached_response["answer"]
            verified_sources = cached_response.get("verified_sources", [])
            top_score = cached_response.get("top_score", 1.0)
            status_label = "CACHE_HIT"
            cache_hit = True
        else:
            cache_hit = False
            # Step 3: Vector Retrieval & Dual-Threshold Gate
            retrieval_res = st.session_state.retriever.retrieve(
                query=search_query,
                vector_store=st.session_state.vector_store,
                k=4,
                max_chunks_per_doc=2
            )

            top_score = retrieval_res["top_score"]
            status_label = retrieval_res["status"]
            passed_chunks = retrieval_res["passed_chunks"]

            if status_label == "REFUSE":
                # Deterministic early exit - 0 LLM calls!
                answer_text = UNIFIED_REFUSAL_MESSAGE
                verified_sources = []
            else:
                # Step 4: Grounded LLM Generation
                with st.spinner("Analyzing context chunks and generating grounded answer..."):
                    prompt = build_grounded_qa_prompt(passed_chunks, user_query)
                    raw_llm_out = ask_llm(prompt)
                    llm_call_count += 1

                    structured_res = parse_structured_response(raw_llm_out, passed_chunks)

                    if not structured_res["answerable"]:
                        answer_text = UNIFIED_REFUSAL_MESSAGE
                        verified_sources = []
                    else:
                        answer_text = structured_res["answer"]
                        verified_sources = structured_res["verified_sources"]

                        if structured_res.get("conflict_detected"):
                            answer_text += f"\n\n⚠️ **Cross-Document Discrepancy Detected:** {structured_res.get('conflict_explanation')}"

                        # Step 5: Store in Semantic Cache
                        st.session_state.semantic_cache.store(
                            query=search_query,
                            collection_hash=st.session_state.collection_hash,
                            response={
                                "answer": answer_text,
                                "verified_sources": verified_sources,
                                "top_score": top_score
                            }
                        )

        # Update Conversation Memory
        st.session_state.conversation.add_turn("user", user_query)
        st.session_state.conversation.add_turn("assistant", answer_text)

        # Display Assistant Response
        with st.chat_message("assistant"):
            st.markdown(answer_text)
            st.caption(
                f"Top Similarity: `{top_score:.3f}` | Status: `{status_label}` | "
                f"Cache Hit: `{cache_hit}` | LLM Invocations: `{llm_call_count}`"
            )

            if verified_sources:
                with st.expander("🔍 Verified Source References & Citations"):
                    for s in verified_sources:
                        st.markdown(f"**[{s['chunk_id']}]** *{s['filename']}* (Page {s['page']})")
                        st.markdown(f"> {s['snippet']}")

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer_text,
            "metrics": {
                "top_score": top_score,
                "status": status_label,
                "cache_hit": cache_hit,
                "llm_calls": llm_call_count
            },
            "sources": verified_sources
        })
