import streamlit as st
from content import load_file
from chunks import chunk_text

st.title("📄 Smart Document Reader (with Chunking)")

uploaded_file = st.file_uploader(
    "Upload your file",
    type=["pdf", "txt", "md", "xls", "xlsx"]
)

if uploaded_file is not None:

    # Step 1: Extract text
    text = load_file(uploaded_file)

    st.subheader("📜 Extracted Text:")
    st.write(text[:1000])

    # Step 2: Chunking
    chunks = chunk_text(text)

    st.subheader("🧩 Chunks:")
    st.write(f"Total chunks: {len(chunks)}")

    # Show sample chunks
    for i, chunk in enumerate(chunks[:3]):
        st.write(f"Chunk {i+1}:")
        st.write(chunk)