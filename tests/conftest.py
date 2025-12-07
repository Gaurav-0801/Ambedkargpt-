from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict

import networkx as nx
import pytest
import yaml


@pytest.fixture(scope="session")
def sample_config(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("ambedkargpt")
    data_dir = base_dir / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = data_dir / "Ambedkar_book.pdf"
    pdf_path.write_text("Sample content", encoding="utf-8")

    chunks_payload: Dict = {
        "stats": {},
        "chunks": [
            {
                "id": "chunk_0",
                "text": "Dr. Ambedkar advocated for social equality.",
                "token_count": 7,
                "sentence_indices": [0],
                "pages": [1],
                "embedding": [1.0, 0.0],
            }
        ],
        "sub_chunks": [
            {
                "id": "chunk_0::sub::0",
                "parent_id": "chunk_0",
                "text": "Dr. Ambedkar advocated for social equality.",
                "token_count": 7,
                "pages": [1],
                "sentence_indices": [0],
                "embedding": [1.0, 0.0],
            }
        ],
    }
    chunks_path = processed_dir / "chunks.json"
    chunks_path.write_text(json.dumps(chunks_payload), encoding="utf-8")

    G = nx.Graph()
    G.add_node(
        "Dr. B. R. Ambedkar",
        label="PERSON",
        count=1,
        chunks=["chunk_0"],
        pages=[1],
        embedding=[1.0, 0.0],
        community=0,
    )
    graph_path = processed_dir / "knowledge_graph.pkl"
    graph_path.write_bytes(pickle.dumps(G))

    partition_path = processed_dir / "community_partition.json"
    partition_path.write_text(
        json.dumps({"partition": {"Dr. B. R. Ambedkar": 0}, "communities": {0: ["Dr. B. R. Ambedkar"]}}),
        encoding="utf-8",
    )

    reports_path = processed_dir / "community_reports.json"
    reports_path.write_text(
        json.dumps(
            {
                "reports": [
                    {
                        "community_id": 0,
                        "summary": "Community discussing Ambedkar's advocacy for equality.",
                        "relations": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    config = {
        "paths": {
            "pdf": str(pdf_path),
            "chunks": str(chunks_path),
            "graph": str(graph_path),
            "node_index": str(processed_dir / "node_index.json"),
            "community_partition": str(partition_path),
            "community_reports": str(reports_path),
        },
        "embeddings": {"sentence_model": "dummy-model", "batch_size": 2},
        "chunking": {
            "threshold": 0.3,
            "buffer_size": 1,
            "max_tokens": 128,
            "sub_chunk_tokens": 32,
            "overlap_tokens": 8,
        },
        "graph": {"min_entity_length": 3, "allow_types": ["PERSON", "ORG"]},
        "retrieval": {
            "tau_entity": 0.1,
            "tau_chunk": 0.1,
            "top_k_entities": 5,
            "top_k_chunks": 5,
            "top_k_communities": 3,
            "top_k_points": 5,
        },
        "llm": {"model": "llama3", "temperature": 0.2, "max_tokens": 64},
    }
    config_path = base_dir / "config.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    return config_path

