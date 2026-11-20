from langchain_text_splitters import RecursiveCharacterTextSplitter


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
