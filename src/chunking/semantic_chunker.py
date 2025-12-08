from __future__ import annotations

import gc
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

from .buffer_merger import buffer_merge

nltk.download("punkt", quiet=True)

console = Console()
app = typer.Typer(no_args_is_help=True)


@dataclass
class SentenceRecord:
    index: int
    text: str
    page: int


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
            if not config:
                raise ValueError(f"Config file is empty or invalid: {path}")
            return config
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML config file {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading config from {path}: {e}") from e


def ensure_processed_dir(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


def load_pdf_sentences(pdf_path: Path) -> List[SentenceRecord]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        reader = PdfReader(str(pdf_path))
        if len(reader.pages) == 0:
            raise ValueError(f"PDF file is empty: {pdf_path}")
        sentences: List[SentenceRecord] = []

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for sent in nltk.sent_tokenize(text):
                cleaned = " ".join(sent.split())
                if not cleaned:
                    continue
                sentences.append(SentenceRecord(index=len(sentences), text=cleaned, page=page_idx))

        if not sentences:
            raise ValueError(f"No sentences extracted from PDF: {pdf_path}")
        return sentences
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise RuntimeError(f"Error loading PDF from {pdf_path}: {e}") from e


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
    # Free memory
    del embeddings
    gc.collect()


def attach_embeddings_in_batches(
    model: SentenceTransformer, entries: List[Dict[str, Any]], batch_size: int, embedding_batch_size: int = None
) -> None:
    """
    Attach embeddings to entries in batches to reduce memory usage.
    Processes entries in smaller batches and frees memory after each batch.
    """
    if embedding_batch_size is None:
        embedding_batch_size = batch_size
    
    total = len(entries)
    for i in range(0, total, embedding_batch_size):
        batch = entries[i:i + embedding_batch_size]
        texts = [entry["text"] for entry in batch]
        embeddings = embed_texts(model, texts, batch_size=batch_size)
        for entry, emb in zip(batch, embeddings):
            entry["embedding"] = emb.tolist()
        # Free memory after each batch
        del embeddings
        gc.collect()


def compute_distances_in_batches(
    model: SentenceTransformer,
    buffered_sentences: List[Dict[str, Any]],
    batch_size: int,
    embedding_batch_size: int = None,
) -> List[float]:
    """
    Compute cosine distances between consecutive buffered sentences in batches.
    This avoids loading all embeddings into memory at once.
    """
    if embedding_batch_size is None:
        embedding_batch_size = min(100, len(buffered_sentences))  # Process 100 at a time
    
    distances: List[float] = []
    total = len(buffered_sentences)
    
    # Process in overlapping windows to compute consecutive distances
    # We need embeddings for pairs (i, i+1), so we process in windows
    i = 0
    prev_embedding = None
    
    while i < total:
        # Process a batch
        end_idx = min(i + embedding_batch_size, total)
        batch_texts = [b["text"] for b in buffered_sentences[i:end_idx]]
        batch_embeddings = embed_texts(model, batch_texts, batch_size=batch_size)
        
        # Compute distances for this batch
        if prev_embedding is not None:
            # Distance between last embedding of previous batch and first of current
            sim = float(np.dot(prev_embedding, batch_embeddings[0]))
            distances.append(1.0 - sim)
        
        # Compute distances within this batch
        if len(batch_embeddings) > 1:
            sims = np.sum(batch_embeddings[:-1] * batch_embeddings[1:], axis=1)
            batch_dists = (1.0 - sims).tolist()
            distances.extend(batch_dists)
        
        # Store last embedding for next batch
        prev_embedding = batch_embeddings[-1].copy()
        
        # Free memory
        del batch_embeddings
        gc.collect()
        
        i = end_idx
    
    return distances


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
    """
    Implements Algorithm 1 (Semantic Chunking) from SEMRAG paper.
    
    Steps:
    1. Extract sentences from PDF
    2. Apply buffer merging to preserve context
    3. Generate embeddings for buffered sentences
    4. Compute cosine distances between consecutive embeddings
    5. Create chunk boundaries when distance exceeds threshold
    6. Split chunks into sub-chunks with overlap for fine-grained retrieval
    """
    try:
        cfg = load_config(config)
        if "paths" not in cfg:
            raise ValueError("Config missing 'paths' section")
        if "chunking" not in cfg:
            raise ValueError("Config missing 'chunking' section")
        if "embeddings" not in cfg:
            raise ValueError("Config missing 'embeddings' section")
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

        # Process buffered embeddings in batches to compute distances
        # This avoids loading all embeddings into memory at once
        embedding_batch_size = emb_cfg.get("embedding_batch_size", 100)
        batch_size = emb_cfg.get("batch_size", 16)
        console.log(f"Computing distances in batches (batch size: {embedding_batch_size})...")
        distances = compute_distances_in_batches(
            model, buffered, batch_size=batch_size, embedding_batch_size=embedding_batch_size
        )

        chunks = split_into_chunks(
            buffered,
            distances,
            tokenizer=tokenizer,
            threshold=chunk_cfg["threshold"],
            max_tokens=chunk_cfg["max_tokens"],
        )
        if not chunks:
            raise ValueError("No chunks created. Check PDF content and chunking parameters.")
        console.log(f"Semantic chunks created: {len(chunks)}")

        # Free memory from buffered data and distances
        del distances
        gc.collect()

        # Process chunk embeddings in batches
        console.log("Attaching embeddings to chunks...")
        attach_embeddings_in_batches(
            model, chunks, batch_size=batch_size, embedding_batch_size=embedding_batch_size
        )

        # Create all sub-chunks first (text only, not memory intensive)
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
        
        # Process sub-chunk embeddings in batches (CRITICAL: this was the main memory issue)
        console.log("Attaching embeddings to sub-chunks (this may take a while)...")
        attach_embeddings_in_batches(
            model, sub_chunks, batch_size=batch_size, embedding_batch_size=embedding_batch_size
        )

        stats = {
            "sentences": len(sentences),
            "buffered_sentences": len(buffered),
            "chunks": len(chunks),
            "sub_chunks": len(sub_chunks),
        }

        output_path = Path(paths_cfg["chunks"])
        persist_chunks(output_path, chunks, sub_chunks, stats)
        console.log(f"Chunks persisted to {output_path}")
        
        # Final memory cleanup
        del buffered
        del sentences
        gc.collect()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error during chunking:[/red] {e}")
        raise


if __name__ == "__main__":
    app()
