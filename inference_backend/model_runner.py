from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from cesnet_models.models import Model_30pktTCNET_256_Weights, model_30pktTCNET_256
from torch import nn
from torch.nn import functional as F

from inference_backend.config import BackendSettings


def _normalize_embeddings(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1)


@dataclass
class LoadedEmbedder:
    model: nn.Module
    ppi_transform: Callable
    output_dim: int


def _replace_unseen_packet_embeddings(model: nn.Module, threshold: int) -> None:
    if threshold <= 0:
        return
    backbone = getattr(model, "backbone_model", None)
    if backbone is None:
        return
    histogram = getattr(backbone, "psizes_hist", None)
    packet_embedding = getattr(backbone, "packet_size_nn_embedding", None)
    if histogram is None or packet_embedding is None:
        return

    counts = np.asarray(histogram)
    if counts.ndim != 1 or counts.shape[0] < 1501:
        return

    unseen = np.where(counts < threshold)[0]
    if unseen.size == 0:
        return

    weight = packet_embedding.weight.data
    # Small packet sizes are sparse and unstable. Map them to padding embedding.
    for pkt_size in unseen[unseen < 100]:
        weight[pkt_size] = weight[0]

    seen_large = [idx for idx in range(1250, 1501) if idx not in unseen]
    if seen_large:
        for pkt_size in unseen[unseen >= 1250]:
            nearest = min(seen_large, key=lambda x: abs(x - pkt_size))
            weight[pkt_size] = weight[nearest]


def load_embedder(settings: BackendSettings, pretrained: bool = True) -> LoadedEmbedder:
    weights = Model_30pktTCNET_256_Weights.DEFAULT if pretrained else None
    model = model_30pktTCNET_256(weights=weights)

    backbone = getattr(model, "backbone_model", None)
    if backbone is not None and hasattr(backbone, "save_psizes_hist"):
        # Histogram tracking is only diagnostic and can fail on CPU with integer histc.
        backbone.save_psizes_hist = False

    if settings.feature_mode == "backbone_gem":
        def _forward_backbone_gem(ppi: torch.Tensor) -> torch.Tensor:
            return model.backbone_model.forward_features(ppi=ppi, flowstats=None)

        model.forward = _forward_backbone_gem  # type: ignore[assignment]

    _replace_unseen_packet_embeddings(model, settings.replace_unseen_packet_threshold)

    model = model.to(settings.device)
    model.eval()

    if settings.compile_model:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
        except Exception:
            # Fall back to eager mode if the runtime cannot compile this model.
            pass

    ppi_transform = (weights.transforms["ppi_transform"] if weights is not None else (lambda x: x))

    dummy = np.zeros((2, 3, 30), dtype=np.float32)
    with torch.inference_mode():
        transformed = ppi_transform(dummy).astype(np.float32, copy=False)
        sample = torch.from_numpy(transformed).to(settings.device)
        sample_out = model(sample)
    output_dim = int(sample_out.shape[1])

    return LoadedEmbedder(model=model, ppi_transform=ppi_transform, output_dim=output_dim)


class EmbeddingModelRunner:
    """Compute embeddings from packet-level inputs with batched inference."""

    def __init__(self, settings: BackendSettings, pretrained: bool = True) -> None:
        self.settings = settings
        loaded = load_embedder(settings=settings, pretrained=pretrained)
        self.model = loaded.model
        self.ppi_transform = loaded.ppi_transform
        self.embedding_dim = loaded.output_dim
        self.device = torch.device(settings.device)

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def _validate_flows(self, flows: np.ndarray) -> np.ndarray:
        if flows.ndim != 3:
            raise ValueError(f"Expected rank-3 array [batch, 3, 30], got shape={flows.shape}")
        if flows.shape[1] != 3:
            raise ValueError(f"Expected second dimension to be 3, got shape={flows.shape}")
        if flows.shape[2] != 30:
            raise ValueError(f"Expected third dimension to be 30, got shape={flows.shape}")
        return flows

    def _prepare(self, flows: np.ndarray) -> np.ndarray:
        flows = np.ascontiguousarray(flows, dtype=np.float32)
        self._validate_flows(flows)
        transformed = self.ppi_transform(flows)
        transformed = np.ascontiguousarray(transformed, dtype=np.float32)
        self._validate_flows(transformed)
        return transformed

    def embed(self, flows: np.ndarray, normalize: bool = True) -> np.ndarray:
        data = self._prepare(flows)
        batch_size = self.settings.batch_size
        outputs: list[torch.Tensor] = []

        with torch.inference_mode():
            for start in range(0, len(data), batch_size):
                stop = min(start + batch_size, len(data))
                batch_np = data[start:stop]
                batch = torch.from_numpy(batch_np).to(self.device, non_blocking=(self.device.type == "cuda"))

                if self.settings.use_amp and self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model(batch)
                else:
                    out = self.model(batch)

                if normalize:
                    out = _normalize_embeddings(out)
                outputs.append(out.detach().cpu())

        return torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)
