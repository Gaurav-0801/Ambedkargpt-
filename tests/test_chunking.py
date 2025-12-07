from __future__ import annotations

from typing import List

import tiktoken

from chunking.semantic_chunker import split_into_chunks, split_with_overlap


def _make_buffered_entry(text: str, idx: int) -> dict:
    return {
        "text": text,
        "sentence_indices": [idx],
        "pages": [1],
        "start_index": idx,
        "end_index": idx,
    }


def test_split_into_chunks_respects_token_threshold():
    tokenizer = tiktoken.get_encoding("cl100k_base")
    buffered = [
        _make_buffered_entry("Ambedkar fought caste discrimination.", 0),
        _make_buffered_entry("He emphasized social justice.", 1),
        _make_buffered_entry("Education was central to his vision.", 2),
    ]

    # Force a low threshold to trigger chunk boundary on every sentence
    distances = [1.0, 1.0]
    chunks = split_into_chunks(buffered, distances, tokenizer, threshold=0.2, max_tokens=20)
    assert len(chunks) == 3
    assert all(chunk["token_count"] <= 20 for chunk in chunks)


def test_split_with_overlap_preserves_context():
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunk = {
        "id": "chunk_0",
        "text": "Ambedkar argued for liberty, equality, and fraternity.",
        "sentence_indices": [0],
        "pages": [1],
    }

    sub_chunks = split_with_overlap(tokenizer, chunk, sub_chunk_tokens=10, overlap_tokens=4)
    assert sub_chunks, "Sub-chunking should produce at least one segment."
    assert sub_chunks[0]["parent_id"] == "chunk_0"
    # successive sub-chunks should overlap on token spans
    for idx in range(1, len(sub_chunks)):
        assert sub_chunks[idx]["start_token"] < sub_chunks[idx - 1]["end_token"]

