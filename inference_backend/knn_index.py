from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("faiss is required for kNN index operations") from exc


@dataclass
class KnnSearchResult:
    scores: np.ndarray
    indices: np.ndarray
    labels: np.ndarray


class FaissKnnIndex:
    """Thread-safe FAISS wrapper supporting cosine and L2 search."""

    def __init__(self, metric: str = "cosine", use_gpu: bool = True) -> None:
        if metric not in {"cosine", "l2"}:
            raise ValueError("metric must be one of: cosine, l2")
        self.metric = metric
        self.use_gpu = use_gpu
        self._labels: np.ndarray | None = None
        self._index = None
        self._gpu_res = None
        self._lock = threading.RLock()

    @property
    def ntotal(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    @property
    def dim(self) -> int | None:
        if self._index is None:
            return None
        return int(self._index.d)

    def _build_empty_index(self, dim: int):
        if self.metric == "cosine":
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)

        if self.use_gpu and hasattr(faiss, "StandardGpuResources"):
            self._gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(self._gpu_res, 0, index)
        return index

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected rank-2 vectors [n, d], got shape={arr.shape}")
        if self.metric == "cosine":
            arr = arr.copy()
            faiss.normalize_L2(arr)
        return arr

    def build(self, embeddings: np.ndarray, labels: np.ndarray | list[str]) -> None:
        with self._lock:
            vectors = self._prepare_vectors(embeddings)
            labels_arr = np.asarray(labels)
            if len(vectors) != len(labels_arr):
                raise ValueError("embeddings and labels must have the same length")

            self._index = self._build_empty_index(vectors.shape[1])
            self._index.add(vectors)  # type: ignore[arg-type]
            self._labels = labels_arr

    def search(self, embeddings: np.ndarray, top_k: int) -> KnnSearchResult:
        with self._lock:
            if self._index is None or self._labels is None:
                raise RuntimeError("kNN index is not initialized")

            queries = self._prepare_vectors(embeddings)
            top_k = max(1, min(int(top_k), self.ntotal))
            scores, indices = self._index.search(queries, top_k)  # type: ignore[arg-type]

            label_matrix: np.ndarray = np.empty(indices.shape, dtype=object)
            label_matrix[:] = None
            valid = indices >= 0
            if np.any(valid):
                label_matrix[valid] = self._labels[indices[valid]]

            return KnnSearchResult(scores=scores, indices=indices, labels=label_matrix)

    def save(self, output_dir: str) -> tuple[str, str, str]:
        with self._lock:
            if self._index is None or self._labels is None:
                raise RuntimeError("kNN index is not initialized")

            os.makedirs(output_dir, exist_ok=True)
            index_path = os.path.join(output_dir, "index.faiss")
            labels_path = os.path.join(output_dir, "labels.npy")
            meta_path = os.path.join(output_dir, "meta.json")

            index_to_save = self._index
            if self.use_gpu and hasattr(faiss, "index_gpu_to_cpu"):
                index_to_save = faiss.index_gpu_to_cpu(index_to_save)

            faiss.write_index(index_to_save, index_path)
            np.save(labels_path, self._labels, allow_pickle=True)

            with open(meta_path, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "metric": self.metric,
                        "dim": int(index_to_save.d),
                        "size": int(index_to_save.ntotal),
                    },
                    fp,
                    indent=2,
                )

            return index_path, labels_path, meta_path

    def load(self, index_path: str, labels_path: str) -> None:
        with self._lock:
            if not os.path.exists(index_path):
                raise FileNotFoundError(index_path)
            if not os.path.exists(labels_path):
                raise FileNotFoundError(labels_path)

            index = faiss.read_index(index_path)
            if self.use_gpu and hasattr(faiss, "StandardGpuResources"):
                self._gpu_res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(self._gpu_res, 0, index)

            labels = np.load(labels_path, allow_pickle=True)

            self._index = index
            self._labels = labels
