from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_bytes
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def load_pdf(file):
    text = ""

    reader = PdfReader(file)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    # for scanned files
    if len(text.strip()) < 50:
        file.seek(0)
        images = convert_from_bytes(file.read(), dpi=300)

        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)

    return text


def load_txt(file):
    return file.read().decode("utf-8")


def load_md(file):
    return file.read().decode("utf-8")


def load_excel(file):
    df = pd.read_excel(file)
    return df.to_csv(index=False)


def load_file(file):
    file_type = file.name.split(".")[-1].lower()

    if file_type == "pdf":
        return load_pdf(file)

    elif file_type == "txt":
        return load_txt(file)

    elif file_type == "md":
        return load_md(file)

    elif file_type in ["xls", "xlsx"]:
        return load_excel(file)

    else:
        return "Unsupported file type"