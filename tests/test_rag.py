import sys
import os
import shutil

# Ensure project root is in sys.path for direct pytest invocation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np

from utils.content import normalize_text, load_file_content
from utils.chunks import chunk_document
from rag.embedder import normalize_l2
from rag.vector import VectorStore
from rag.retriever import Retriever
from llm.model import parse_structured_response
from llm.conversation import ConversationManager
from llm.cache import SemanticCache


def test_normalize_text():
    dirty = "Line 1\r\n\r\n\r\nLine 2   with   spaces\r\n"
    clean = normalize_text(dirty)
    assert "\r" not in clean
    assert "Line 1\n\nLine 2 with spaces" == clean


def test_load_file_content_txt():
    doc_data = load_file_content("sample_docs/doc_a_os_scheduling.txt", "doc_a_os_scheduling.txt")
    assert doc_data["filename"] == "doc_a_os_scheduling.txt"
    assert len(doc_data["pages"]) == 1
    assert doc_data["total_chars"] > 100
    assert doc_data["error"] is None


def test_load_scanned_pdf_warning():
    doc_data = load_file_content("sample_docs/doc_d_unreadable_scan.pdf", "doc_d_unreadable_scan.pdf")
    assert doc_data["warning"] is not None
    assert "scanned" in doc_data["warning"].lower()


def test_chunk_document_metadata():
    doc_data = load_file_content("sample_docs/doc_c_process_memory.txt", "doc_c_process_memory.txt")
    chunks = chunk_document(doc_data, chunk_size=800, chunk_overlap=150)
    assert len(chunks) > 0

    first = chunks[0]
    assert "chunk_id" in first
    assert first["doc_id"] == "doc_c_process_memory_txt"
    assert first["page"] == 1
    assert "content" in first
    assert len(first["content"]) <= 950


def test_vector_store_l2_and_duplicates():
    store = VectorStore(dim=4)
    raw_vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.01, 0.0, 0.0],  # Near duplicate of #1 (cos sim > 0.99)
        [0.0, 1.0, 0.0, 0.0]   # Orthogonal (cos sim = 0.0)
    ], dtype=np.float32)

    norm_vecs = normalize_l2(raw_vecs)
    chunks = [
        {"chunk_id": "c1", "content": "Chunk 1", "doc_id": "d1"},
        {"chunk_id": "c2", "content": "Chunk 2 (dup)", "doc_id": "d1"},
        {"chunk_id": "c3", "content": "Chunk 3", "doc_id": "d1"}
    ]

    added = store.add(norm_vecs, chunks, duplicate_threshold=0.97)
    assert added == 2  # c2 should have been suppressed as duplicate
    assert store.count() == 2


def test_vector_store_persistence(tmp_path):
    store = VectorStore(dim=4)
    vecs = normalize_l2(np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32))
    chunks = [
        {"chunk_id": "c1", "content": "Text 1", "filename": "f1.txt"},
        {"chunk_id": "c2", "content": "Text 2", "filename": "f2.txt"}
    ]
    store.add(vecs, chunks)

    save_dir = str(tmp_path / "faiss_test")
    store.save(save_dir)

    loaded = VectorStore.load(save_dir)
    assert loaded.count() == 2
    assert loaded.metadata[0]["chunk_id"] == "c1"


def test_retriever_hard_floor_gate():
    store = VectorStore(dim=4)
    # Target vector pointing in X direction
    vecs = normalize_l2(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    chunks = [{"chunk_id": "c1", "content": "Operating System", "doc_id": "d1"}]
    store.add(vecs, chunks)

    retriever = Retriever(hard_floor=0.30, confident_threshold=0.50)

    # Completely orthogonal query (dot product = 0.0 < 0.30)
    scores, results = store.search(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert scores[0] < 0.30


def test_parse_structured_response_filters_unretrieved_ids():
    retrieved = [
        {"chunk_id": "valid_chunk_01", "filename": "doc1.txt", "page": 1, "content": "Real info"}
    ]
    raw_llm_json = """{
        "answerable": true,
        "answer": "This is an answer [valid_chunk_01] and [fake_chunk_99].",
        "cited_chunk_ids": ["valid_chunk_01", "fake_chunk_99"],
        "conflict_detected": false
    }"""

    parsed = parse_structured_response(raw_llm_json, retrieved)
    assert parsed["answerable"] is True
    # fake_chunk_99 must be rejected!
    assert "fake_chunk_99" not in parsed["cited_chunk_ids"]
    assert "valid_chunk_01" in parsed["cited_chunk_ids"]
    assert len(parsed["verified_sources"]) == 1


def test_conversation_manager_heuristics():
    cm = ConversationManager()
    # No history -> standalone
    assert cm.requires_reformulation("What is deadlock?") is False

    cm.add_turn("user", "What is deadlock?")
    cm.add_turn("assistant", "Deadlock is a state of waiting.")

    # Short query (< 6 words) in active session -> triggers reformulation
    assert cm.requires_reformulation("Why?") is True
    assert cm.requires_reformulation("What about Doc B?") is True
    assert cm.requires_reformulation("Explain it more") is True

    # Detailed, long standalone query -> bypasses reformulation
    assert cm.requires_reformulation("What are the primary differences between paging and segmentation in memory systems?") is False
