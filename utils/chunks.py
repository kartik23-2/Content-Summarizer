from langchain_text_splitters import RecursiveCharacterTextSplitter


def simple_chunk(text, chunk_size=500, overlap=100):
    """
    Upgraded chunking using RecursiveCharacterTextSplitter
    while keeping same function signature.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",   # paragraph
            "\n",     # line
            ".",      # sentence
            " ",      # word
            ""        # fallback
        ]
    )

    chunks = splitter.split_text(text)

    return chunks