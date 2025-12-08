from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import yaml
from rich.console import Console
from sentence_transformers import SentenceTransformer

from src.graph.entity_extractor import analyze_text

console = Console()


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


def load_chunks(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}. Run 'python -m src.pipeline.ambedkargpt chunk' first.")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if "chunks" not in data:
                raise ValueError(f"Invalid chunks file format: missing 'chunks' key in {path}")
            return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON chunks file {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading chunks from {path}: {e}") from e


def normalize_entity(text: str) -> str:
    return " ".join(text.split())


def build_knowledge_graph(
    chunks: List[Dict[str, Any]],
    graph_cfg: Dict[str, Any],
) -> nx.Graph:
    G = nx.Graph()
    node_metadata: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "label": "",
            "chunks": set(),
            "pages": set(),
            "examples": set(),
        }
    )

    for chunk in chunks:
        chunk_id = chunk["id"]
        text = chunk["text"]
        pages = chunk.get("pages", [])

        entities, relations = analyze_text(
            text,
            allowed_types=graph_cfg.get("allow_types"),
            min_length=graph_cfg.get("min_entity_length", 3),
        )

        for entity in entities:
            name = normalize_entity(entity["text"])
            node_metadata[name]["count"] += 1
            node_metadata[name]["label"] = entity["label"]
            node_metadata[name]["chunks"].add(chunk_id)
            node_metadata[name]["pages"].update(pages)
            if len(node_metadata[name]["examples"]) < 5:
                node_metadata[name]["examples"].add(text[:300])

        for rel in relations:
            src = normalize_entity(rel["source"])
            tgt = normalize_entity(rel["target"])
            if src == tgt:
                continue
            relation_label = rel["relation"]
            evidence = rel["sentence"]

            if G.has_edge(src, tgt):
                data = G[src][tgt]
                data["weight"] += 1
                data["relations"].add(relation_label)
                data["evidence"].append({"chunk_id": chunk_id, "text": evidence})
            else:
                G.add_edge(
                    src,
                    tgt,
                    weight=1,
                    relations={relation_label},
                    evidence=[{"chunk_id": chunk_id, "text": evidence}],
                )

    for entity, meta in node_metadata.items():
        G.add_node(
            entity,
            label=meta["label"],
            count=meta["count"],
            chunks=sorted(meta["chunks"]),
            pages=sorted(meta["pages"]),
            examples=list(meta["examples"]),
        )

    # Convert relation sets back to lists for serialization safety
    for u, v, data in G.edges(data=True):
        data["relations"] = list(data["relations"])

    return G


def attach_entity_embeddings(
    graph: nx.Graph,
    model: SentenceTransformer,
    batch_size: int,
) -> None:
    entity_names = list(graph.nodes())
    if not entity_names:
        return

    embeddings = model.encode(
        entity_names,
        batch_size=batch_size,
        show_progress_bar=len(entity_names) > batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    for name, emb in zip(entity_names, embeddings):
        graph.nodes[name]["embedding"] = emb.tolist()


def persist_graph(graph: nx.Graph, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(graph, fh)
    except PermissionError as e:
        raise PermissionError(f"Permission denied writing to {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error persisting graph to {path}: {e}") from e


def write_node_index(graph: nx.Graph, path: Path) -> None:
    index = []
    for node, data in graph.nodes(data=True):
        index.append(
            {
                "entity": node,
                "label": data.get("label"),
                "count": data.get("count"),
                "pages": data.get("pages", []),
                "chunks": data.get("chunks", []),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"entities": index}, fh, indent=2)


def main(config_path: Path = Path("config.yaml")) -> None:
    try:
        cfg = load_config(config_path)
        if "graph" not in cfg:
            raise ValueError("Config missing 'graph' section")
        if "paths" not in cfg:
            raise ValueError("Config missing 'paths' section")
        graph_cfg = cfg["graph"]
        paths_cfg = cfg["paths"]
        emb_cfg = cfg.get("embeddings", {})

        console.rule("[bold]Knowledge Graph Builder[/bold]")
        chunk_data = load_chunks(Path(paths_cfg["chunks"]))
        chunks = chunk_data["chunks"]
        if not chunks:
            raise ValueError(f"No chunks found in {paths_cfg['chunks']}. Ensure chunking step completed successfully.")
        console.log(f"Loaded {len(chunks)} chunks from {paths_cfg['chunks']}")

        graph = build_knowledge_graph(chunks, graph_cfg)
        if graph.number_of_nodes() == 0:
            console.print("[yellow]Warning:[/yellow] No entities extracted from chunks. Graph is empty.")
        console.log(f"Graph nodes: {graph.number_of_nodes()} edges: {graph.number_of_edges()}")

        model = SentenceTransformer(cfg["embeddings"]["sentence_model"])
        attach_entity_embeddings(graph, model, batch_size=emb_cfg.get("batch_size", 16))

        graph_path = Path(paths_cfg["graph"])
        persist_graph(graph, graph_path)
        console.log(f"Graph persisted to {graph_path}")

        node_index_path = Path(paths_cfg["node_index"])
        write_node_index(graph, node_index_path)
        console.log(f"Node index written to {node_index_path}")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise


if __name__ == "__main__":
    main()
