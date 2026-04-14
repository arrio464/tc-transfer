from __future__ import annotations

import argparse
import gc
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
    parser = argparse.ArgumentParser(
        description="kNN transfer and index artifact export"
    )
    parser.add_argument(
        "--dataset-npz",
        required=True,
        help="Path to NPZ with train/val/test flows and labels",
    )
    parser.add_argument("--output-dir", required=True, help="Artifact output directory")

    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--feature-mode", default="backbone_gem", choices=["original", "backbone_gem"]
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--amp", action="store_true", help="Enable mixed precision on CUDA"
    )
    parser.add_argument(
        "--finetune-checkpoint",
        default=None,
        help="Optional checkpoint produced by post_training.finetune_full (best_finetune.pt)",
    )
    parser.add_argument(
        "--strict-paper-knn",
        action="store_true",
        help="Enforce paper-like kNN protocol: fixed pretrained embeddings + cosine + top1",
    )

    parser.add_argument("--metric", default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--k-candidates",
        default="1,3,5,9",
        help="Comma-separated list for automatic K tuning on val split",
    )
    parser.add_argument("--auto-tune-k", action="store_true")
    parser.add_argument(
        "--vote",
        default="top1",
        choices=["top1", "majority"],
        help="Prediction strategy",
    )
    parser.add_argument(
        "--selection-metric",
        default="macro_recall",
        choices=["top1_acc", "macro_recall", "weighted_f1"],
        help="Metric used when --auto-tune-k is enabled",
    )

    parser.add_argument("--faiss-gpu", action="store_true")
    parser.add_argument(
        "--clear-cuda-cache-between-phases",
        action="store_true",
        help="Call torch.cuda.empty_cache() between major phases to reduce peak VRAM",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _maybe_clear_cuda_cache(device: torch.device, enabled: bool) -> None:
    if not enabled:
        return
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _compute_embeddings(
    model,
    ppi_transform,
    flows: np.ndarray,
    device: torch.device,
    batch_size: int,
    amp: bool,
    output_dim: int,
    desc: str,
) -> np.ndarray:
    flows = np.asarray(flows, dtype=np.float32)
    if len(flows) == 0:
        return np.zeros((0, output_dim), dtype=np.float32)

    outputs = np.empty((len(flows), output_dim), dtype=np.float32)

    with torch.inference_mode():
        for start in tqdm(range(0, len(flows), batch_size), desc=desc):
            stop = min(start + batch_size, len(flows))
            batch_np = np.ascontiguousarray(
                ppi_transform(flows[start:stop]), dtype=np.float32
            )
            batch = torch.from_numpy(batch_np).to(
                device, non_blocking=(device.type == "cuda")
            )
            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(batch)
            else:
                out = model(batch)
            outputs[start:stop] = (
                out.detach().cpu().numpy().astype(np.float32, copy=False)
            )

    return outputs


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


def _evaluate_with_k(
    index: FaissKnnIndex,
    embeddings: np.ndarray,
    labels: np.ndarray,
    top_k: int,
    vote: str,
) -> dict[str, float]:
    result = index.search(embeddings=embeddings, top_k=top_k)
    pred = _predict_labels(result.labels, vote=vote)
    return classification_metrics(labels, pred)


def _load_finetune_checkpoint(
    model: torch.nn.Module, checkpoint_path: str, device: torch.device
) -> dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Unsupported checkpoint format: expected dict payload")

    if "model_state" in payload:
        model.load_state_dict(payload["model_state"], strict=True)
    else:
        # Allow loading raw state-dict checkpoints as a fallback.
        model.load_state_dict(payload, strict=True)
    return payload


def _enforce_protocol_best_practices(args: argparse.Namespace) -> None:
    if args.strict_paper_knn:
        if args.finetune_checkpoint:
            raise ValueError("--strict-paper-knn does not allow --finetune-checkpoint")
        if args.metric != "cosine":
            raise ValueError("--strict-paper-knn requires --metric cosine")
        if args.vote != "top1":
            raise ValueError("--strict-paper-knn requires --vote top1")
        if args.auto_tune_k:
            raise ValueError("--strict-paper-knn does not allow --auto-tune-k")
        if args.top_k != 1:
            raise ValueError("--strict-paper-knn requires --top-k 1")

    # For top1 voting, k>1 does not affect predictions and can be misleading.
    if args.vote == "top1" and args.auto_tune_k:
        print(
            "[warn] --auto-tune-k has no effect with --vote top1; disabling auto tuning"
        )
        args.auto_tune_k = False

    if args.vote == "top1" and args.top_k != 1:
        print("[warn] --vote top1 ignores neighbors beyond rank 1; forcing top-k to 1")
        args.top_k = 1


def main() -> None:
    args = parse_args()
    _enforce_protocol_best_practices(args)

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

    checkpoint_meta: dict | None = None
    if args.finetune_checkpoint:
        checkpoint_meta = _load_finetune_checkpoint(
            model=model, checkpoint_path=args.finetune_checkpoint, device=device
        )
        print(f"Loaded finetune checkpoint: {args.finetune_checkpoint}")
        if isinstance(checkpoint_meta, dict):
            ckpt_feature_mode = checkpoint_meta.get("feature_mode")
            if (
                ckpt_feature_mode is not None
                and str(ckpt_feature_mode) != settings.feature_mode
            ):
                print(
                    f"[warn] checkpoint feature_mode={ckpt_feature_mode} differs from requested feature_mode={settings.feature_mode}; "
                    "embedding behavior may differ"
                )

    splits = load_npz_splits(args.dataset_npz)

    if len(splits.train_flows) == 0:
        raise RuntimeError("Training split is empty; cannot build kNN index")
    if len(splits.train_flows) != len(splits.train_labels):
        raise RuntimeError("Training flows/labels length mismatch")
    if len(splits.val_flows) != len(splits.val_labels):
        raise RuntimeError("Validation flows/labels length mismatch")
    if len(splits.test_flows) != len(splits.test_labels):
        raise RuntimeError("Test flows/labels length mismatch")

    started = time.time()

    train_embeddings = _compute_embeddings(
        model=model,
        ppi_transform=ppi_transform,
        flows=splits.train_flows,
        device=device,
        batch_size=args.batch_size,
        amp=args.amp,
        output_dim=loaded.output_dim,
        desc="Embedding train",
    )

    index = FaissKnnIndex(metric=args.metric, use_gpu=args.faiss_gpu)
    index.build(embeddings=train_embeddings, labels=splits.train_labels)
    train_index_size = int(index.ntotal)
    del train_embeddings
    _maybe_clear_cuda_cache(device=device, enabled=args.clear_cuda_cache_between_phases)

    k_candidates = [int(v.strip()) for v in args.k_candidates.split(",") if v.strip()]
    k_candidates = [k for k in k_candidates if k > 0]
    if not k_candidates:
        k_candidates = [args.top_k]
    k_candidates = sorted(set(k_candidates))

    selected_k = max(1, args.top_k)
    val_scores = {}

    if args.auto_tune_k and len(splits.val_flows) > 0 and len(splits.val_labels) > 0:
        val_embeddings = _compute_embeddings(
            model=model,
            ppi_transform=ppi_transform,
            flows=splits.val_flows,
            device=device,
            batch_size=args.batch_size,
            amp=args.amp,
            output_dim=loaded.output_dim,
            desc="Embedding val",
        )
        best = -1.0
        for k in k_candidates:
            metrics = _evaluate_with_k(
                index=index,
                embeddings=val_embeddings,
                labels=splits.val_labels,
                top_k=k,
                vote=args.vote,
            )
            score = metrics[args.selection_metric]
            val_scores[str(k)] = metrics
            if score > best:
                best = score
                selected_k = k
        del val_embeddings
        _maybe_clear_cuda_cache(
            device=device, enabled=args.clear_cuda_cache_between_phases
        )
    else:
        selected_k = max(1, args.top_k)

    test_embeddings = _compute_embeddings(
        model=model,
        ppi_transform=ppi_transform,
        flows=splits.test_flows,
        device=device,
        batch_size=args.batch_size,
        amp=args.amp,
        output_dim=loaded.output_dim,
        desc="Embedding test",
    )

    requested_k = selected_k
    selected_k = max(1, min(selected_k, train_index_size))
    if selected_k != requested_k:
        print(
            f"[warn] requested top-k={requested_k} exceeds index size={train_index_size}; using top-k={selected_k}"
        )

    test_metrics = _evaluate_with_k(
        index=index,
        embeddings=test_embeddings,
        labels=splits.test_labels,
        top_k=selected_k,
        vote=args.vote,
    )
    del test_embeddings
    _maybe_clear_cuda_cache(device=device, enabled=args.clear_cuda_cache_between_phases)

    index_dir = os.path.join(args.output_dir, "knn_index")
    index_path, labels_path, meta_path = index.save(index_dir)

    save_json(
        os.path.join(args.output_dir, "metrics.json"),
        {
            "selected_k": selected_k,
            "requested_k": requested_k,
            "selection_metric": args.selection_metric,
            "test_metrics": test_metrics,
            "val_scores": val_scores,
            "protocol": "paper" if args.strict_paper_knn else "deployment",
            "split_sizes": {
                "train": int(len(splits.train_labels)),
                "val": int(len(splits.val_labels)),
                "test": int(len(splits.test_labels)),
            },
            "index_artifact": {
                "index_path": index_path,
                "labels_path": labels_path,
                "meta_path": meta_path,
                "metric": args.metric,
            },
            "source_checkpoint": args.finetune_checkpoint,
            "checkpoint_meta": {
                "epoch": checkpoint_meta.get("epoch")
                if isinstance(checkpoint_meta, dict)
                else None,
                "selection_metric": checkpoint_meta.get("selection_metric")
                if isinstance(checkpoint_meta, dict)
                else None,
                "best_val_metric": checkpoint_meta.get("best_val_metric")
                if isinstance(checkpoint_meta, dict)
                else None,
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
