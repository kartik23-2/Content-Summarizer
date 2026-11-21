import os
import re
import unicodedata
import warnings

try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
except ImportError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from PyPDF2 import PdfReader
        from PyPDF2.errors import PdfReadError


def normalize_text(text: str) -> str:
    """Normalize unicode, collapse whitespace, and strip carriage returns."""
    if not text:
        return ""
    # Unicode NFKD normalization
    text = unicodedata.normalize("NFKD", text)
    # Replace carriage returns and excessive whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple inline spaces/tabs to a single space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_file_content(file_obj_or_path, filename: str) -> dict:
    """
    Load and parse a document (PDF or TXT) into page-level text.
    Handles corrupt, empty, and image-only/scanned files gracefully.
    
    Returns:
    {
        "filename": filename,
        "doc_id": safe_doc_id,
        "pages": [{"page_number": int, "text": str}],
        "total_chars": int,
        "warning": str | None,
        "error": str | None
    }
    """
    safe_doc_id = re.sub(r"[^a-zA-Z0-9_]", "_", filename).lower()
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    result = {
        "filename": filename,
        "doc_id": safe_doc_id,
        "pages": [],
        "total_chars": 0,
        "warning": None,
        "error": None
    }

    try:
        if ext == "pdf":
            try:
                reader = PdfReader(file_obj_or_path)
            except PdfReadError as pe:
                result["error"] = f"Corrupt or encrypted PDF: {pe}"
                return result
            except Exception as e:
                result["error"] = f"Failed to parse PDF: {e}"
                return result

            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    result["error"] = "Password-protected PDF cannot be read without password."
                    return result

            for page_idx, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                clean_page = normalize_text(page_text)
                result["pages"].append({
                    "page_number": page_idx,
                    "text": clean_page
                })
                result["total_chars"] += len(clean_page)

            # Check for low-text / scanned PDF
            avg_chars = result["total_chars"] / max(len(result["pages"]), 1)
            if result["total_chars"] < 50 or avg_chars < 30:
                result["warning"] = (
                    f"Warning: '{filename}' contains very little extractable text "
                    f"({result['total_chars']} characters across {len(result['pages'])} pages). "
                    "The document appears to be scanned or image-based."
                )

        elif ext in ["txt", "md"]:
            content = ""
            if hasattr(file_obj_or_path, "read"):
                raw_bytes = file_obj_or_path.read()
                if isinstance(raw_bytes, str):
                    content = raw_bytes
                else:
                    content = raw_bytes.decode("utf-8", errors="ignore")
            elif isinstance(file_obj_or_path, str) and os.path.exists(file_obj_or_path):
                with open(file_obj_or_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

            clean_text = normalize_text(content)
            result["pages"].append({
                "page_number": 1,
                "text": clean_text
            })
            result["total_chars"] = len(clean_text)

            if result["total_chars"] == 0:
                result["warning"] = f"Warning: '{filename}' is empty."

        else:
            result["error"] = f"Unsupported file format: '.{ext}'. Supported formats are PDF and TXT."

    except Exception as e:
        result["error"] = f"Unexpected error reading '{filename}': {str(e)}"

    return result
