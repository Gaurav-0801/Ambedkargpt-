from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

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


def load_reports(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("reports", [])


class GlobalGraphRAG:
    def __init__(self, config_path: Path = Path("config.yaml")) -> None:
        cfg = load_config(config_path)
        self.paths = cfg["paths"]
        self.retrieval_cfg = cfg["retrieval"]
        self.emb_cfg = cfg.get("embeddings", {})

        self.graph = load_graph(Path(self.paths["graph"]))
        chunk_data = load_chunks(Path(self.paths["chunks"]))
        self.sub_chunks = chunk_data["sub_chunks"]
        self.reports = load_reports(Path(self.paths["community_reports"]))

        self.model = SentenceTransformer(cfg["embeddings"]["sentence_model"])

        self.community_embeddings = self._prepare_community_embeddings()
        self.community_chunks = self._map_community_chunks()

    def _prepare_community_embeddings(self) -> Dict[int, np.ndarray]:
        embeddings = {}
        for report in self.reports:
            community_id = report["community_id"]
            text = report["summary"] + "\n" + "\n".join(report.get("relations", []))
            emb = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embeddings[community_id] = emb
        return embeddings

    def _map_community_chunks(self) -> Dict[int, List[Dict[str, Any]]]:
        mapping: Dict[int, List[Dict[str, Any]]] = {}
        parent_to_sub = {}
        for sub_chunk in self.sub_chunks:
            parent_to_sub.setdefault(sub_chunk.get("parent_id"), []).append(sub_chunk)

        for _, data in self.graph.nodes(data=True):
            community_id = data.get("community")
            if community_id is None:
                continue
            mapping.setdefault(community_id, [])
            for chunk_id in data.get("chunks", []):
                    mapping[community_id].extend(parent_to_sub.get(chunk_id, []))

        return mapping

    def _top_communities(self, query_vec: np.ndarray) -> List[int]:
        scores = []
        for community_id, emb in self.community_embeddings.items():
            score = float(np.dot(query_vec, emb))
            scores.append((community_id, score))
        if not scores:
            return []
        scores.sort(key=lambda item: item[1], reverse=True)
        return [cid for cid, _ in scores[: self.retrieval_cfg["top_k_communities"]]]

    def _rank_points(self, query_vec: np.ndarray, community_id: int) -> List[Dict[str, Any]]:
        points = []
        for sub_chunk in self.community_chunks.get(community_id, []):
            emb = sub_chunk.get("embedding")
            if not emb:
                continue
            score = float(np.dot(query_vec, to_vector(emb)))
            points.append(
                {
                    "community_id": community_id,
                    "chunk_id": sub_chunk["id"],
                    "parent_id": sub_chunk["parent_id"],
                    "text": sub_chunk["text"],
                    "score": score,
                    "pages": sub_chunk.get("pages", []),
                }
            )

        points.sort(key=lambda item: item["score"], reverse=True)
        return points[: self.retrieval_cfg["top_k_points"]]

    def search(self, query: str) -> List[Dict[str, Any]]:
        query_vec = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        community_ids = self._top_communities(query_vec)
        results: List[Dict[str, Any]] = []

        for cid in community_ids:
            points = self._rank_points(query_vec, cid)
            if not points:
                continue
            results.append({"community_id": cid, "points": points})

        return results


def global_graph_rag_search(
    query: str,
    config_path: Path = Path("config.yaml"),
) -> List[Dict[str, Any]]:
    retriever = GlobalGraphRAG(config_path=config_path)
    return retriever.search(query)
