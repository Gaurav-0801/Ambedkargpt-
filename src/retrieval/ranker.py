from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def to_vector(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.ndim != 1 or vec_b.ndim != 1:
        raise ValueError("cosine_similarity expects 1D vectors")
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) or 1e-12
    return float(np.dot(vec_a, vec_b) / denom)


def batch_similarity(query_vec: np.ndarray, vectors: Iterable[np.ndarray]) -> List[float]:
    return [float(np.dot(query_vec, vec)) for vec in vectors]


def top_k(items: Sequence[Tuple[float, dict]], k: int) -> List[Tuple[float, dict]]:
    return sorted(items, key=lambda x: x[0], reverse=True)[:k]

