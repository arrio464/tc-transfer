from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from inference_backend.config import BackendSettings
from inference_backend.knn_index import FaissKnnIndex
from inference_backend.model_runner import EmbeddingModelRunner


@dataclass
class BackendStatus:
    device: str
    feature_mode: str
    embedding_dim: int
    knn_ready: bool
    knn_size: int
    knn_metric: str


class InferenceRuntime:
    def __init__(self, settings: BackendSettings) -> None:
        self.settings = settings
        self.embedder = EmbeddingModelRunner(settings=settings, pretrained=True)
        self.index = FaissKnnIndex(metric=settings.index_metric, use_gpu=settings.use_gpu_faiss)

        if settings.startup_index_path and settings.startup_labels_path:
            self.index.load(index_path=settings.startup_index_path, labels_path=settings.startup_labels_path)

    def status(self) -> dict:
        return asdict(
            BackendStatus(
                device=self.settings.device,
                feature_mode=self.settings.feature_mode,
                embedding_dim=self.embedder.embedding_dim,
                knn_ready=self.index.ntotal > 0,
                knn_size=self.index.ntotal,
                knn_metric=self.settings.index_metric,
            )
        )

    def compute_embeddings(self, flows: np.ndarray, normalize: bool = True) -> tuple[np.ndarray, float]:
        start = perf_counter()
        embeddings = self.embedder.embed(flows=flows, normalize=normalize)
        elapsed_ms = (perf_counter() - start) * 1000.0
        return embeddings, elapsed_ms

    def load_index(self, index_path: str, labels_path: str) -> dict:
        self.index.load(index_path=index_path, labels_path=labels_path)
        return {
            "index_size": self.index.ntotal,
            "index_dim": self.index.dim,
            "metric": self.settings.index_metric,
        }

    def search_by_embedding(self, embeddings: np.ndarray, top_k: int) -> tuple[dict, float]:
        start = perf_counter()
        result = self.index.search(embeddings, top_k=top_k)
        elapsed_ms = (perf_counter() - start) * 1000.0

        neighbors: list[list[dict]] = []
        for row_idx in range(result.indices.shape[0]):
            row = []
            for col_idx in range(result.indices.shape[1]):
                idx = int(result.indices[row_idx, col_idx])
                label = result.labels[row_idx, col_idx]
                score = float(result.scores[row_idx, col_idx])
                row.append(
                    {
                        "index": idx,
                        "label": None if label is None else str(label),
                        "score": score,
                    }
                )
            neighbors.append(row)

        payload = {
            "neighbors": neighbors,
            "top_k": int(result.indices.shape[1]),
            "num_queries": int(result.indices.shape[0]),
        }
        return payload, elapsed_ms

    def predict_knn(self, flows: np.ndarray, top_k: int) -> tuple[dict, float]:
        start = perf_counter()
        embeddings = self.embedder.embed(flows=flows, normalize=True)
        result = self.index.search(embeddings, top_k=top_k)

        predictions = []
        neighbors: list[list[dict]] = []
        for row_idx in range(result.indices.shape[0]):
            labels = [
                None if label is None else str(label)
                for label in result.labels[row_idx].tolist()
            ]
            valid_labels = [label for label in labels if label is not None]
            if not valid_labels:
                pred = None
            elif top_k == 1:
                pred = valid_labels[0]
            else:
                counts = Counter(valid_labels)
                max_count = max(counts.values())
                tied = {label for label, count in counts.items() if count == max_count}
                pred = next(label for label in valid_labels if label in tied)
            predictions.append(pred)

            row = []
            for col_idx in range(result.indices.shape[1]):
                row.append(
                    {
                        "index": int(result.indices[row_idx, col_idx]),
                        "label": labels[col_idx],
                        "score": float(result.scores[row_idx, col_idx]),
                    }
                )
            neighbors.append(row)

        elapsed_ms = (perf_counter() - start) * 1000.0
        payload = {
            "predictions": predictions,
            "neighbors": neighbors,
            "top_k": int(result.indices.shape[1]),
            "num_samples": int(result.indices.shape[0]),
        }
        return payload, elapsed_ms
