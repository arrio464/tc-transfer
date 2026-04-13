from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch

ALLOWED_FEATURE_MODES = {"original", "backbone_gem"}
ALLOWED_INDEX_METRICS = {"cosine", "l2"}


def _read_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _read_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class BackendSettings:
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    log_level: str = "info"

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    feature_mode: str = "backbone_gem"
    batch_size: int = 4096
    compile_model: bool = False
    use_amp: bool = True
    replace_unseen_packet_threshold: int = 1

    default_top_k: int = 5
    max_top_k: int = 100
    index_metric: str = "cosine"
    use_gpu_faiss: bool = True

    startup_index_path: str | None = None
    startup_labels_path: str | None = None
    startup_meta_path: str | None = None

    @staticmethod
    def from_env() -> "BackendSettings":
        settings = BackendSettings(
            host=os.getenv("TC_BACKEND_HOST", "0.0.0.0"),
            port=_read_int(os.getenv("TC_BACKEND_PORT"), 8080),
            workers=max(1, _read_int(os.getenv("TC_BACKEND_WORKERS"), 1)),
            log_level=os.getenv("TC_BACKEND_LOG_LEVEL", "info"),
            device=os.getenv("TC_BACKEND_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"),
            feature_mode=os.getenv("TC_BACKEND_FEATURE_MODE", "backbone_gem"),
            batch_size=max(1, _read_int(os.getenv("TC_BACKEND_BATCH_SIZE"), 4096)),
            compile_model=_read_bool(os.getenv("TC_BACKEND_COMPILE_MODEL"), False),
            use_amp=_read_bool(os.getenv("TC_BACKEND_USE_AMP"), True),
            replace_unseen_packet_threshold=max(0, _read_int(os.getenv("TC_BACKEND_REPLACE_UNSEEN_THRESHOLD"), 1)),
            default_top_k=max(1, _read_int(os.getenv("TC_BACKEND_DEFAULT_TOP_K"), 5)),
            max_top_k=max(1, _read_int(os.getenv("TC_BACKEND_MAX_TOP_K"), 100)),
            index_metric=os.getenv("TC_BACKEND_INDEX_METRIC", "cosine"),
            use_gpu_faiss=_read_bool(os.getenv("TC_BACKEND_USE_GPU_FAISS"), True),
            startup_index_path=os.getenv("TC_BACKEND_INDEX_PATH"),
            startup_labels_path=os.getenv("TC_BACKEND_LABELS_PATH"),
            startup_meta_path=os.getenv("TC_BACKEND_INDEX_META_PATH"),
        )
        return settings.validated()

    def validated(self) -> "BackendSettings":
        feature_mode = self.feature_mode.strip().lower()
        if feature_mode not in ALLOWED_FEATURE_MODES:
            raise ValueError(f"Unsupported feature_mode: {self.feature_mode}. Supported: {sorted(ALLOWED_FEATURE_MODES)}")

        metric = self.index_metric.strip().lower()
        if metric not in ALLOWED_INDEX_METRICS:
            raise ValueError(f"Unsupported index_metric: {self.index_metric}. Supported: {sorted(ALLOWED_INDEX_METRICS)}")

        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k cannot be greater than max_top_k")

        return replace(self, feature_mode=feature_mode, index_metric=metric)
