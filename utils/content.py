from PyPDF2 import PdfReader
import requests
import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def get_ocr_api_key():
    try:
        return st.secrets.get("OCR_API_KEY") or os.getenv("OCR_API_KEY")
    except Exception:
        return os.getenv("OCR_API_KEY")

def extract_text_with_ocr(file):
    ocr_api_key = get_ocr_api_key()
    if not ocr_api_key:
        st.warning("OCR_API_KEY is missing. OCR for scanned PDFs/images may return empty text.")
        return ""

    file.seek(0)

    try:
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": file},
            data={
                "apikey": ocr_api_key,
                "language": "eng"
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result.get("ParsedResults", [{}])[0].get("ParsedText", "") or ""
    except Exception:
        return ""
    
def load_pdf(file):
    text = ""

    # Try extracting normal text
    reader = PdfReader(file)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    # If no text → use OCR API
    if len(text.strip()) < 50:
        text = extract_text_with_ocr(file)

    return text

def load_image(file):
    return extract_text_with_ocr(file)


def load_txt(file):
    return file.read().decode("utf-8", errors="ignore")


def load_excel(file):
    df = pd.read_excel(file)
    return df.to_string(index=False)


def load_csv(file):
    df = pd.read_csv(file)
    return df.to_string(index=False)


def load_file(file):
    file_type = file.name.split(".")[-1].lower()

    if file_type == "pdf":
        return load_pdf(file)

    elif file_type == "txt":
        return load_txt(file)

    elif file_type in ["xls", "xlsx"]:
        return load_excel(file)

    elif file_type == "csv":
        return load_csv(file)
    
    elif file_type in ["jpg", "jpeg", "png", "webp"]:
        return load_image(file)

    else:
        return "Unsupported file type"