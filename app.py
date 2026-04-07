import streamlit as st
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_bytes


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.title(" Smart PDF Reader (Text + OCR)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    
   #for normal pdf
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    # for scanned pdf
    if len(text.strip()) < 50:
        st.warning(" Using OCR (scanned PDF detected)")
        
        uploaded_file.seek(0)  
        pdf_bytes = uploaded_file.read()

        images = convert_from_bytes(pdf_bytes)
        text = ""

        for img in images:
            text += pytesseract.image_to_string(img)

    #  result
    st.subheader(" Extracted Text:")
    st.write(text[:2000])