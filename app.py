import streamlit as st
from content import load_file

st.title(" Smart Document Reader (PDF + OCR + Multi-format)")

uploaded_file = st.file_uploader(
    "Upload your file",
    type=["pdf", "txt", "md", "xls", "xlsx"]
)

if uploaded_file is not None:

    text = load_file(uploaded_file)

    st.subheader(" Extracted Text:")
    st.write(text[:2000])

    st.write(f"Total characters: {len(text)}")