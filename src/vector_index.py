"""In-memory vector index with cosine-similarity search."""
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SearchHit:
    text: str
    score: float
    metadata: dict[str, Any]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # all-zero vectors stay zero and score 0 against everything
    return matrix / norms


class InMemoryVectorIndex:
    """Stores L2-normalised vectors so cosine similarity is one matrix product."""

    def __init__(self, embedder):
        self.embedder = embedder
        self._texts: list[str] = []
        self._meta: list[dict[str, Any]] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._texts)

    def add(self, texts: list[str], metadata: list[dict[str, Any]] | None = None) -> None:
        if not texts:
            return
        metadata = metadata if metadata is not None else [{} for _ in texts]
        if len(metadata) != len(texts):
            raise ValueError("metadata must have the same length as texts")
        vectors = _normalize(np.asarray(self.embedder.embed(texts), dtype=np.float32))
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        self._texts.extend(texts)
        self._meta.extend(metadata)

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        if self._matrix is None or k <= 0:
            return []
        q = _normalize(np.asarray(self.embedder.embed([query]), dtype=np.float32))[0]
        scores = self._matrix @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [SearchHit(self._texts[i], float(scores[i]), self._meta[i]) for i in order]
