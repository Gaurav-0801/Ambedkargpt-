from __future__ import annotations

import numpy as np
import pytest

from retrieval.global_search import GlobalGraphRAG
from retrieval.local_search import LocalGraphRAG


class DummyModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            if "equality" in text.lower():
                vectors.append(np.array([1.0, 0.0], dtype=np.float32))
            else:
                vectors.append(np.array([0.0, 1.0], dtype=np.float32))
        return np.vstack(vectors)


@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch):
    monkeypatch.setattr(
        "retrieval.local_search.SentenceTransformer",
        lambda *args, **kwargs: DummyModel(),
    )
    monkeypatch.setattr(
        "retrieval.global_search.SentenceTransformer",
        lambda *args, **kwargs: DummyModel(),
    )


def test_local_graph_rag_search_returns_ranked_chunks(sample_config):
    retriever = LocalGraphRAG(config_path=sample_config)
    results = retriever.search("What did Ambedkar say about equality?")
    assert results, "Local search should return at least one entity."
    assert results[0]["chunks"], "Entity should include supporting chunks."
    assert results[0]["chunks"][0]["chunk_id"].startswith("chunk_0")


def test_global_graph_rag_search_returns_points(sample_config):
    retriever = GlobalGraphRAG(config_path=sample_config)
    results = retriever.search("Tell me about equality.")
    assert results, "Global search should return communities."
    assert results[0]["points"], "Community should include scored points."
    assert results[0]["points"][0]["chunk_id"].startswith("chunk_0")

