from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import nltk
import numpy as np
import tiktoken
import typer
import yaml
from pypdf import PdfReader
from rich.console import Console
from rich.progress import track
from sentence_transformers import SentenceTransformer

from chunking.buffer_merger import buffer_merge

nltk.download("punkt", quiet=True)

console = Console()
app = typer.Typer(no_args_is_help=True)


@dataclass
class SentenceRecord:
    index: int
    text: str
    page: int


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_processed_dir(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


def load_pdf_sentences(pdf_path: Path) -> List[SentenceRecord]:
    reader = PdfReader(str(pdf_path))
    sentences: List[SentenceRecord] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for sent in nltk.sent_tokenize(text):
            cleaned = " ".join(sent.split())
            if not cleaned:
                continue
            sentences.append(SentenceRecord(index=len(sentences), text=cleaned, page=page_idx))

    return sentences


def prepare_buffered_sentences(sentences: List[SentenceRecord], buffer_size: int) -> List[Dict[str, Any]]:
    serializable = [{"text": s.text, "page": s.page, "index": s.index} for s in sentences]
    return buffer_merge(serializable, buffer_size)


def embed_texts(
    model: SentenceTransformer, texts: Sequence[str], batch_size: int = 16
) -> np.ndarray:
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()))
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=len(texts) > batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def cosine_distances(embeddings: np.ndarray) -> List[float]:
    if len(embeddings) < 2:
        return []
    sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    dists = 1 - sims
    return dists.tolist()


def sentence_token_length(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def make_chunk_record(
    idx: int,
    entries: List[Dict[str, Any]],
    tokenizer,
) -> Dict[str, Any]:
    text = " ".join(e["text"] for e in entries).strip()
    sentence_indices = sorted({i for entry in entries for i in entry["sentence_indices"]})
    pages = sorted({p for entry in entries for p in entry.get("pages", [])})
    token_count = sentence_token_length(tokenizer, text)
    return {
        "id": f"chunk_{idx}",
        "text": text,
        "token_count": token_count,
        "sentence_indices": sentence_indices,
        "pages": pages,
    }


def split_into_chunks(
    buffered_sentences: List[Dict[str, Any]],
    distances: List[float],
    tokenizer,
    threshold: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    current_tokens = 0
    chunk_idx = 0

    for idx, entry in enumerate(buffered_sentences):
        entry_tokens = sentence_token_length(tokenizer, entry["text"])

        if current and current_tokens + entry_tokens > max_tokens:
            chunks.append(make_chunk_record(chunk_idx, current, tokenizer))
            chunk_idx += 1
            current = []
            current_tokens = 0

        current.append(entry)
        current_tokens += entry_tokens

        is_boundary = idx == len(buffered_sentences) - 1
        if not is_boundary and idx < len(distances):
            is_boundary = distances[idx] >= threshold

        if is_boundary:
            chunks.append(make_chunk_record(chunk_idx, current, tokenizer))
            chunk_idx += 1
            current = []
            current_tokens = 0

    if current:
        chunks.append(make_chunk_record(chunk_idx, current, tokenizer))

    # Deduplicate empty chunks
    return [c for c in chunks if c["text"]]


def split_with_overlap(
    tokenizer,
    chunk: Dict[str, Any],
    sub_chunk_tokens: int,
    overlap_tokens: int,
) -> List[Dict[str, Any]]:
    tokens = tokenizer.encode(chunk["text"])
    if not tokens:
        return []

    sub_chunks: List[Dict[str, Any]] = []
    start = 0
    step = max(sub_chunk_tokens - overlap_tokens, 1)
    idx = 0

    while start < len(tokens):
        end = min(len(tokens), start + sub_chunk_tokens)
        sub_tokens = tokens[start:end]
        sub_text = tokenizer.decode(sub_tokens).strip()
        if sub_text:
            sub_chunks.append(
                {
                    "id": f"{chunk['id']}::sub::{idx}",
                    "parent_id": chunk["id"],
                    "text": sub_text,
                    "token_count": len(sub_tokens),
                    "start_token": start,
                    "end_token": end,
                    "sentence_indices": chunk["sentence_indices"],
                    "pages": chunk["pages"],
                }
            )
            idx += 1

        if end == len(tokens):
            break
        start = max(0, end - overlap_tokens)

    return sub_chunks


def attach_embeddings(
    model: SentenceTransformer, entries: List[Dict[str, Any]], batch_size: int
) -> None:
    texts = [entry["text"] for entry in entries]
    embeddings = embed_texts(model, texts, batch_size=batch_size)
    for entry, emb in zip(entries, embeddings):
        entry["embedding"] = emb.tolist()


def persist_chunks(
    output_path: Path,
    chunks: List[Dict[str, Any]],
    sub_chunks: List[Dict[str, Any]],
    stats: Dict[str, Any],
) -> None:
    ensure_processed_dir(output_path)
    payload = {
        "stats": stats,
        "chunks": chunks,
        "sub_chunks": sub_chunks,
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


@app.command()
def run(config: Path = typer.Option(Path("config.yaml"), "--config", "-c")) -> None:
    cfg = load_config(config)
    paths_cfg = cfg["paths"]
    chunk_cfg = cfg["chunking"]
    emb_cfg = cfg.get("embeddings", {})

    tokenizer = tiktoken.get_encoding("cl100k_base")
    model = SentenceTransformer(cfg["embeddings"]["sentence_model"])

    console.rule("[bold]Semantic Chunking[/bold]")
    pdf_path = Path(paths_cfg["pdf"])
    sentences = load_pdf_sentences(pdf_path)
    console.log(f"Loaded {len(sentences)} sentences from {pdf_path}")

    buffered = prepare_buffered_sentences(sentences, chunk_cfg["buffer_size"])
    console.log(f"Buffered sentences count: {len(buffered)}")

    buffered_embeddings = embed_texts(model, [b["text"] for b in buffered], batch_size=emb_cfg.get("batch_size", 16))
    distances = cosine_distances(buffered_embeddings)

    chunks = split_into_chunks(
        buffered,
        distances,
        tokenizer=tokenizer,
        threshold=chunk_cfg["threshold"],
        max_tokens=chunk_cfg["max_tokens"],
    )
    console.log(f"Semantic chunks created: {len(chunks)}")

    attach_embeddings(model, chunks, batch_size=emb_cfg.get("batch_size", 16))

    sub_chunks: List[Dict[str, Any]] = []
    for chunk in track(chunks, description="Creating sub-chunks"):
        sub_chunks.extend(
            split_with_overlap(
                tokenizer=tokenizer,
                chunk=chunk,
                sub_chunk_tokens=chunk_cfg["sub_chunk_tokens"],
                overlap_tokens=chunk_cfg["overlap_tokens"],
            )
        )

    console.log(f"Sub-chunks generated: {len(sub_chunks)}")
    attach_embeddings(model, sub_chunks, batch_size=emb_cfg.get("batch_size", 16))

    stats = {
        "sentences": len(sentences),
        "buffered_sentences": len(buffered),
        "chunks": len(chunks),
        "sub_chunks": len(sub_chunks),
    }

    output_path = Path(paths_cfg["chunks"])
    persist_chunks(output_path, chunks, sub_chunks, stats)
    console.log(f"Chunks persisted to {output_path}")


if __name__ == "__main__":
    app()
