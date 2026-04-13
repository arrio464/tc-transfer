from __future__ import annotations

import argparse
import os
import time
from collections import Counter

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

from inference_backend.config import BackendSettings
from inference_backend.knn_index import FaissKnnIndex
from inference_backend.model_runner import load_embedder
from post_training.common import (
    classification_metrics,
    load_npz_splits,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kNN transfer and index artifact export")
    parser.add_argument("--dataset-npz", required=True, help="Path to NPZ with train/val/test flows and labels")
    parser.add_argument("--output-dir", required=True, help="Artifact output directory")

    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-mode", default="backbone_gem", choices=["original", "backbone_gem"])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")

    parser.add_argument("--metric", default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--k-candidates", default="1,3,5,9", help="Comma-separated list for automatic K tuning on val split")
    parser.add_argument("--auto-tune-k", action="store_true")
    parser.add_argument("--vote", default="top1", choices=["top1", "majority"], help="Prediction strategy")

    parser.add_argument("--faiss-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _compute_embeddings(model, ppi_transform, flows: np.ndarray, device: torch.device, batch_size: int, amp: bool) -> np.ndarray:
    flows = np.asarray(flows, dtype=np.float32)
    transformed = ppi_transform(flows).astype(np.float32)
    outputs = []

    with torch.inference_mode():
        for start in tqdm(range(0, len(transformed), batch_size), desc="Embedding"):
            stop = min(start + batch_size, len(transformed))
            batch = torch.from_numpy(transformed[start:stop]).to(device, non_blocking=(device.type == "cuda"))
            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(batch)
            else:
                out = model(batch)
            outputs.append(out.detach().cpu().numpy())

    return np.concatenate(outputs, axis=0).astype(np.float32)


def _predict_labels(neighbor_labels: np.ndarray, vote: str) -> np.ndarray:
    if vote == "top1":
        return np.asarray([row[0] for row in neighbor_labels], dtype=object)

    preds = []
    for row in neighbor_labels:
        row_list = [str(v) for v in row if v is not None]
        if not row_list:
            preds.append(None)
            continue
        counts = Counter(row_list)
        best_count = max(counts.values())
        tied = {label for label, count in counts.items() if count == best_count}
        pred = next(label for label in row_list if label in tied)
        preds.append(pred)
    return np.asarray(preds, dtype=object)


def _evaluate_with_k(index: FaissKnnIndex, embeddings: np.ndarray, labels: np.ndarray, top_k: int, vote: str) -> dict[str, float]:
    result = index.search(embeddings=embeddings, top_k=top_k)
    pred = _predict_labels(result.labels, vote=vote)
    return classification_metrics(labels, pred)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    settings = BackendSettings(
        device=args.device,
        feature_mode=args.feature_mode,
        batch_size=args.batch_size,
        compile_model=False,
        use_amp=args.amp,
        index_metric=args.metric,
        use_gpu_faiss=args.faiss_gpu,
    ).validated()

    loaded = load_embedder(settings=settings, pretrained=True)
    model = loaded.model
    ppi_transform = loaded.ppi_transform

    device = torch.device(settings.device)
    splits = load_npz_splits(args.dataset_npz)

    started = time.time()

    train_embeddings = _compute_embeddings(model, ppi_transform, splits.train_flows, device, args.batch_size, args.amp)
    val_embeddings = _compute_embeddings(model, ppi_transform, splits.val_flows, device, args.batch_size, args.amp) if len(splits.val_flows) > 0 else np.zeros((0, loaded.output_dim), dtype=np.float32)
    test_embeddings = _compute_embeddings(model, ppi_transform, splits.test_flows, device, args.batch_size, args.amp)

    index = FaissKnnIndex(metric=args.metric, use_gpu=args.faiss_gpu)
    index.build(embeddings=train_embeddings, labels=splits.train_labels)

    k_candidates = [int(v.strip()) for v in args.k_candidates.split(",") if v.strip()]
    k_candidates = [k for k in k_candidates if k > 0]
    if not k_candidates:
        k_candidates = [args.top_k]

    selected_k = max(1, args.top_k)
    val_scores = {}

    if args.auto_tune_k and len(val_embeddings) > 0 and len(splits.val_labels) > 0:
        best = -1.0
        for k in k_candidates:
            metrics = _evaluate_with_k(index=index, embeddings=val_embeddings, labels=splits.val_labels, top_k=k, vote=args.vote)
            score = metrics["macro_recall"]
            val_scores[str(k)] = metrics
            if score > best:
                best = score
                selected_k = k
    else:
        selected_k = max(1, args.top_k)

    test_metrics = _evaluate_with_k(
        index=index,
        embeddings=test_embeddings,
        labels=splits.test_labels,
        top_k=selected_k,
        vote=args.vote,
    )

    index_dir = os.path.join(args.output_dir, "knn_index")
    index_path, labels_path, meta_path = index.save(index_dir)

    save_json(
        os.path.join(args.output_dir, "metrics.json"),
        {
            "selected_k": selected_k,
            "test_metrics": test_metrics,
            "val_scores": val_scores,
            "index_artifact": {
                "index_path": index_path,
                "labels_path": labels_path,
                "meta_path": meta_path,
                "metric": args.metric,
            },
            "elapsed_sec": time.time() - started,
            "args": vars(args),
        },
    )

    print("kNN transfer complete")
    print(f"Selected K: {selected_k}")
    print(f"Test metrics: {test_metrics}")
    print(f"Index artifacts saved in: {index_dir}")


if __name__ == "__main__":
    main()
