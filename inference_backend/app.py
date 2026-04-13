from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from starlette.concurrency import run_in_threadpool

from inference_backend.config import BackendSettings
from inference_backend.runtime import InferenceRuntime
from inference_backend.schemas import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    IndexLoadRequest,
    IndexLoadResponse,
    KnnPredictRequest,
    KnnPredictResponse,
    KnnSearchByEmbeddingRequest,
    KnnSearchByEmbeddingResponse,
)


def create_app(settings: BackendSettings | None = None) -> FastAPI:
    settings = settings or BackendSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = InferenceRuntime(settings=settings)
        yield

    app = FastAPI(
        title="TC Transfer Inference Backend",
        version="1.0.0",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        runtime: InferenceRuntime = app.state.runtime
        return HealthResponse(status="ok", details=runtime.status())

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(req: EmbedRequest) -> EmbedResponse:
        runtime: InferenceRuntime = app.state.runtime
        flows = np.asarray(req.flows, dtype=np.float32)
        try:
            embeddings, elapsed_ms = await run_in_threadpool(runtime.compute_embeddings, flows, req.normalize)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            shape=(int(embeddings.shape[0]), int(embeddings.shape[1])),
            elapsed_ms=float(elapsed_ms),
        )

    @app.post("/knn/index/load", response_model=IndexLoadResponse)
    async def knn_index_load(req: IndexLoadRequest) -> IndexLoadResponse:
        runtime: InferenceRuntime = app.state.runtime
        try:
            payload = await run_in_threadpool(runtime.load_index, req.index_path, req.labels_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Failed to load index: {exc}") from exc
        return IndexLoadResponse(message="Index loaded", **payload)

    @app.post("/knn/search_by_embedding", response_model=KnnSearchByEmbeddingResponse)
    async def knn_search_by_embedding(req: KnnSearchByEmbeddingRequest) -> KnnSearchByEmbeddingResponse:
        runtime: InferenceRuntime = app.state.runtime
        embeddings = np.asarray(req.embeddings, dtype=np.float32)
        top_k = req.top_k or settings.default_top_k
        top_k = min(top_k, settings.max_top_k)

        try:
            payload, elapsed_ms = await run_in_threadpool(runtime.search_by_embedding, embeddings, top_k)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return KnnSearchByEmbeddingResponse(elapsed_ms=float(elapsed_ms), **payload)

    @app.post("/knn/search", response_model=KnnSearchByEmbeddingResponse, deprecated=True)
    async def knn_search_deprecated(req: KnnSearchByEmbeddingRequest) -> KnnSearchByEmbeddingResponse:
        # Backward-compatible alias for old clients.
        return await knn_search_by_embedding(req)

    @app.post("/predict/knn", response_model=KnnPredictResponse)
    async def predict_knn(req: KnnPredictRequest) -> KnnPredictResponse:
        runtime: InferenceRuntime = app.state.runtime
        flows = np.asarray(req.flows, dtype=np.float32)
        top_k = req.top_k or settings.default_top_k
        top_k = min(top_k, settings.max_top_k)

        try:
            payload, elapsed_ms = await run_in_threadpool(runtime.predict_knn, flows, top_k)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return KnnPredictResponse(elapsed_ms=float(elapsed_ms), **payload)

    return app
