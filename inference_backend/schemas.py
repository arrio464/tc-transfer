from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    details: dict


class EmbedRequest(BaseModel):
    flows: list[list[list[float]]] = Field(
        ...,
        description="Packet-level input with shape [batch, 3, 30]",
    )
    normalize: bool = Field(default=True, description="L2-normalize output embeddings")


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    shape: tuple[int, int]
    elapsed_ms: float


class KnnSearchByEmbeddingRequest(BaseModel):
    embeddings: list[list[float]]
    top_k: int | None = Field(default=None, ge=1)


class KnnSearchByEmbeddingResponse(BaseModel):
    neighbors: list[list[dict]]
    top_k: int
    num_queries: int
    elapsed_ms: float


class KnnPredictRequest(BaseModel):
    flows: list[list[list[float]]] = Field(
        ...,
        description="Packet-level input with shape [batch, 3, 30]",
    )
    top_k: int | None = Field(default=None, ge=1)


class KnnPredictResponse(BaseModel):
    predictions: list[str | None]
    neighbors: list[list[dict]]
    top_k: int
    num_samples: int
    elapsed_ms: float


class IndexLoadRequest(BaseModel):
    index_path: str
    labels_path: str


class IndexLoadResponse(BaseModel):
    message: str
    index_size: int
    index_dim: int | None
    metric: str
