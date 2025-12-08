from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from src.retrieval.ranker import to_vector


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


def load_graph(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}. Run 'python -m src.pipeline.ambedkargpt build-graph' first.")
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except pickle.UnpicklingError as e:
        raise ValueError(f"Failed to unpickle graph file {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading graph from {path}: {e}") from e


def load_chunks(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}. Run 'python -m src.pipeline.ambedkargpt chunk' first.")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if "sub_chunks" not in data:
                raise ValueError(f"Invalid chunks file format: missing 'sub_chunks' key in {path}")
            return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON chunks file {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading chunks from {path}: {e}") from e


def build_sub_chunk_lookup(sub_chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {chunk["id"]: chunk for chunk in sub_chunks}


def build_parent_index(sub_chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    parent_map: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in sub_chunks:
        parent_map.setdefault(chunk.get("parent_id"), []).append(chunk)
    return parent_map


class LocalGraphRAG:
    """
    Implements Local Graph RAG Search (Equation 4 from SEMRAG paper).
    
    Equation 4: D_retrieved = Top_k({v ∈ V, g ∈ G | sim(v, Q+H) > τ_e ∧ sim(g, v) > τ_d})
    
    Where:
    - v: entities in graph
    - g: chunks associated with entities
    - Q: query
    - H: query history (optional)
    - τ_e: entity similarity threshold (tau_entity)
    - τ_d: chunk similarity threshold (tau_chunk)
    """
    def __init__(self, config_path: Path = Path("config.yaml")) -> None:
        try:
            cfg = load_config(config_path)
            if "paths" not in cfg:
                raise ValueError("Config missing 'paths' section")
            if "retrieval" not in cfg:
                raise ValueError("Config missing 'retrieval' section")
            if "embeddings" not in cfg:
                raise ValueError("Config missing 'embeddings' section")
            self.paths = cfg["paths"]
            self.retrieval_cfg = cfg["retrieval"]
            self.emb_cfg = cfg.get("embeddings", {})

            self.graph = load_graph(Path(self.paths["graph"]))
            chunk_data = load_chunks(Path(self.paths["chunks"]))
            if not chunk_data.get("sub_chunks"):
                raise ValueError(f"No sub_chunks found in {self.paths['chunks']}")
            self.sub_chunks = build_sub_chunk_lookup(chunk_data["sub_chunks"])
            self.parent_index = build_parent_index(chunk_data["sub_chunks"])

            self.model = SentenceTransformer(cfg["embeddings"]["sentence_model"])
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Error initializing LocalGraphRAG: {e}") from e

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
