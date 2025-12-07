from __future__ import annotations

from typing import Any, Dict, List


def buffer_merge(sentences: List[Dict[str, Any]], buffer_size: int) -> List[Dict[str, Any]]:
    """
    Merge neighboring sentences to preserve local context prior to chunking.

    Each merged record keeps track of the source sentence indices and page numbers so the
    downstream pipeline can cite provenance information.
    """

    if buffer_size <= 0:
        return [
            {
                "text": s["text"],
                "sentence_indices": [s["index"]],
                "pages": [s.get("page")] if s.get("page") is not None else [],
                "start_index": s["index"],
                "end_index": s["index"],
            }
            for s in sentences
        ]

    merged: List[Dict[str, Any]] = []
    total = len(sentences)

    for idx in range(total):
        start = max(0, idx - buffer_size)
        end = min(total, idx + buffer_size + 1)
        window = sentences[start:end]

        text = " ".join(item["text"] for item in window).strip()
        if not text:
            continue

        indices = sorted({item["index"] for item in window})
        pages = sorted({item.get("page") for item in window if item.get("page") is not None})

        merged.append(
            {
                "text": text,
                "sentence_indices": indices,
                "pages": pages,
                "start_index": indices[0],
                "end_index": indices[-1],
            }
        )

    return merged

