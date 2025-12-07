from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from retrieval.ranker import to_vector


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_graph(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_chunks(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_sub_chunk_lookup(sub_chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {chunk["id"]: chunk for chunk in sub_chunks}


def build_parent_index(sub_chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    parent_map: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in sub_chunks:
        parent_map.setdefault(chunk.get("parent_id"), []).append(chunk)
    return parent_map


class LocalGraphRAG:
    def __init__(self, config_path: Path = Path("config.yaml")) -> None:
        cfg = load_config(config_path)
        self.paths = cfg["paths"]
        self.retrieval_cfg = cfg["retrieval"]
        self.emb_cfg = cfg.get("embeddings", {})

        self.graph = load_graph(Path(self.paths["graph"]))
        chunk_data = load_chunks(Path(self.paths["chunks"]))
        self.sub_chunks = build_sub_chunk_lookup(chunk_data["sub_chunks"])
        self.parent_index = build_parent_index(chunk_data["sub_chunks"])

        self.model = SentenceTransformer(cfg["embeddings"]["sentence_model"])

    def _entity_candidates(self, query_vec: np.ndarray) -> List[Dict[str, Any]]:
        candidates = []
        tau_e = self.retrieval_cfg["tau_entity"]

        for node, data in self.graph.nodes(data=True):
            embedding = data.get("embedding")
            if not embedding:
                continue
            vec = to_vector(embedding)
            score = float(np.dot(query_vec, vec))
            if score >= tau_e:
                candidates.append(
                    {
                        "entity": node,
                        "score": score,
                        "chunks": data.get("chunks", []),
                        "pages": data.get("pages", []),
                        "label": data.get("label"),
                    }
                )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[: self.retrieval_cfg["top_k_entities"]]

    def _rank_chunks(self, query_vec: np.ndarray, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        tau_chunk = self.retrieval_cfg["tau_chunk"]
        results = []

        for chunk_id in chunk_ids:
            for sub_chunk in self.parent_index.get(chunk_id, []):
                embedding = sub_chunk.get("embedding")
                if not embedding:
                    continue
                score = float(np.dot(query_vec, to_vector(embedding)))
                if score < tau_chunk:
                    continue
                results.append(
                    {
                        "chunk_id": sub_chunk["id"],
                        "parent_id": chunk_id,
                        "score": score,
                        "text": sub_chunk["text"],
                        "pages": sub_chunk.get("pages", []),
                        "sentence_indices": sub_chunk.get("sentence_indices", []),
                    }
                )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[: self.retrieval_cfg["top_k_chunks"]]

    def search(self, query: str, history: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        augmented_query = query
        if history:
            augmented_query = f"{query}\nHistory: {' '.join(history)}"

        query_vec = self.model.encode(
            augmented_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        entities = self._entity_candidates(query_vec)
        results: List[Dict[str, Any]] = []

        for entity in entities:
            chunk_matches = self._rank_chunks(query_vec, entity["chunks"])
            if not chunk_matches:
                continue
            results.append(
                {
                    "entity": entity["entity"],
                    "entity_score": entity["score"],
                    "entity_label": entity.get("label"),
                    "pages": entity.get("pages", []),
                    "chunks": chunk_matches,
                }
            )

        return results


def local_graph_rag_search(
    query: str,
    history: Optional[List[str]] = None,
    config_path: Path = Path("config.yaml"),
) -> List[Dict[str, Any]]:
    retriever = LocalGraphRAG(config_path=config_path)
    return retriever.search(query, history=history)
