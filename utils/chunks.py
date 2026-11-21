try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    class RecursiveCharacterTextSplitter:
        """Standalone fallback recursive character text splitter."""
        def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150, separators: list[str] = None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        def split_text(self, text: str) -> list[str]:
            if not text:
                return []
            
            # Recursive splitting
            def _split(txt: str, seps: list[str]) -> list[str]:
                if not seps or len(txt) <= self.chunk_size:
                    return [txt] if txt else []
                
                sep = seps[0]
                remaining_seps = seps[1:]
                
                if sep == "":
                    # Character level split
                    return [txt[i:i + self.chunk_size] for i in range(0, len(txt), self.chunk_size - self.chunk_overlap)]
                
                splits = txt.split(sep)
                result = []
                for s in splits:
                    if len(s) > self.chunk_size:
                        result.extend(_split(s, remaining_seps))
                    elif s:
                        result.append(s)
                return result

            raw_pieces = _split(text, self.separators)
            
            # Merge into chunks respecting chunk_size and chunk_overlap
            merged = []
            current = ""
            for piece in raw_pieces:
                if not current:
                    current = piece
                elif len(current) + len(piece) + 1 <= self.chunk_size:
                    current += " " + piece
                else:
                    merged.append(current.strip())
                    # Overlap retention
                    overlap_start = max(0, len(current) - self.chunk_overlap)
                    current = current[overlap_start:] + " " + piece

            if current.strip():
                merged.append(current.strip())

            return merged


def chunk_document(doc_data: dict, chunk_size: int = 800, chunk_overlap: int = 150) -> list[dict]:
    """
    Splits document pages into structured chunks preserving metadata.
    Chunk size: 800 characters (~150-180 words)
    Chunk overlap: 150 characters (~30 words)
    
    Returns list of dicts:
    [
        {
            "chunk_id": "doc_id_p1_c01",
            "doc_id": str,
            "filename": str,
            "page": int,
            "char_start": int,
            "char_end": int,
            "content": str
        },
        ...
    ]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",   # Paragraph breaks
            "\n",     # Line breaks
            ". ",     # Sentence boundaries
            "? ",
            "! ",
            " ",      # Words
            ""        # Fallback character
        ]
    )

    chunks = []
    chunk_counter = 1

    for page in doc_data.get("pages", []):
        page_num = page.get("page_number", 1)
        page_text = page.get("text", "").strip()

        if not page_text:
            continue

        raw_splits = splitter.split_text(page_text)

        running_offset = 0
        for split in raw_splits:
            clean_split = split.strip()
            if not clean_split:
                continue

            # Approximate char offsets on this page
            start_pos = page_text.find(clean_split[:50], running_offset)
            if start_pos == -1:
                start_pos = running_offset
            end_pos = start_pos + len(clean_split)
            running_offset = max(running_offset, end_pos - chunk_overlap)

            chunk_id = f"{doc_data['doc_id']}_p{page_num}_c{chunk_counter:02d}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_data["doc_id"],
                "filename": doc_data["filename"],
                "page": page_num,
                "char_start": start_pos,
                "char_end": end_pos,
                "content": clean_split
            })
            chunk_counter += 1

    return chunks


def simple_chunk(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Legacy helper for simple string chunking."""
    mock_doc = {
        "doc_id": "doc",
        "filename": "document.txt",
        "pages": [{"page_number": 1, "text": text}]
    }
    return [c["content"] for c in chunk_document(mock_doc, chunk_size, overlap)]
