"""Embedding creation.

Two interchangeable embedders share one interface, ``embed(texts) -> ndarray``:

* ``FastEmbedEmbedder`` - real semantic embeddings via the FastEmbed library
  (ONNX, no PyTorch). Downloads the model on first use.
* ``HashingEmbedder`` - deterministic hashed bag-of-words vectors. Works fully
  offline and is used for tests. It is lexical, NOT semantic: it will not know
  that "ML" and "machine learning" are related.
"""
import re
import zlib

import numpy as np

DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbedder:
    name = "fastembed"

    def __init__(self, model_name: str = DEFAULT_FASTEMBED_MODEL):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "fastembed is not installed. Run: pip install -r requirements-rag.txt"
            ) from exc
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(list(self._model.embed(texts)), dtype=np.float32)


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9][a-z0-9+#.-]*", text.lower()):
                h = zlib.crc32(token.encode("utf-8"))  # stable across runs, unlike hash()
                out[row, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return out


def get_embedder(name: str = "fastembed"):
    if name == "fastembed":
        return FastEmbedEmbedder()
    if name == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedder: {name!r} (choose 'fastembed' or 'hashing')")
