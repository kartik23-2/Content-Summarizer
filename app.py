import streamlit as st
from PyPDF2 import PdfReader

st.title("📄 PDF Text Extractor")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    # Extract text from each page
    for page in reader.pages:
        text += page.extract_text()

    # Show extracted text
    st.subheader("📜 Extracted Text:")
    st.write(text[:2000])  # show first 2000 characters