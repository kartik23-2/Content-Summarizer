import streamlit as st

from utils.content import load_file
from utils.chunks import simple_chunk

from rag.embedder import get_embeddings
from rag.vector import VectorStore
from rag.retriever import retrieve

from llm.model import ask_llm
from llm.prompt import summary_prompt, qa_prompt

st.set_page_config(page_title="AI Study Assistant", layout="wide")

st.title(" AI Content Summarizer & Q&A Assistant")

uploaded_file = st.file_uploader(
    "Upload document",
    type=["pdf", "txt", "csv", "xls", "xlsx", "jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("processed_file_key") != file_key:
        raw_text = load_file(uploaded_file)

        if not raw_text or not raw_text.strip():
            st.error("Could not extract text from this file.")
            st.stop()

        with st.spinner("Generating summary..."):
            summary = ask_llm(summary_prompt(raw_text))

        chunks = simple_chunk(raw_text)
        if not chunks:
            st.error("No text chunks generated from this file.")
            st.stop()

        embeddings = get_embeddings(chunks)

        # Works for list, tuple, and numpy array
        is_empty_embeddings = (
            embeddings is None
            or (hasattr(embeddings, "size") and embeddings.size == 0)
            or (hasattr(embeddings, "__len__") and len(embeddings) == 0)
        )
        if is_empty_embeddings:
            st.error("Could not generate embeddings.")
            st.stop()

        embedding_dim = embeddings.shape[1] if hasattr(embeddings, "shape") else len(embeddings[0])
        store = VectorStore(embedding_dim)
        store.add(embeddings, chunks)

        st.session_state.processed_file_key = file_key
        st.session_state.raw_text = raw_text
        st.session_state.summary = summary
        st.session_state.store = store

    st.subheader(" Summary")
    st.write(st.session_state.summary)
    st.success(" Document processed! Ready to interact")

    store = st.session_state.store

    query = st.text_input(" Ask a question about the document")

    if query:
        contexts = retrieve(query, store)
        context_text = "\n\n".join(contexts)

        with st.spinner("Thinking..."):
            answer = ask_llm(qa_prompt(context_text, query))

        st.subheader(" Answer")
        st.write(answer)

        st.subheader(" Context Used")
        for c in contexts:
            st.write(c)