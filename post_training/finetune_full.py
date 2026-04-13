from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class _TqdmShim:
        def __init__(self, iterable, *args, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *args, **kwargs):
            return None

    def tqdm(iterable, *args, **kwargs):
        return _TqdmShim(iterable, *args, **kwargs)

from inference_backend.config import BackendSettings
from inference_backend.model_runner import load_embedder
from post_training.common import (
    FlowDataset,
    classification_metrics,
    encode_labels,
    load_npz_splits,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-parameter finetuning for downstream TC tasks")
    parser.add_argument("--dataset-npz", required=True, help="Path to NPZ with train/val/test flows and labels")
    parser.add_argument("--output-dir", required=True, help="Artifact output directory")

    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-mode", default="backbone_gem", choices=["original", "backbone_gem"])
    parser.add_argument("--replace-unseen-threshold", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    head.eval()

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=(device.type == "cuda"))
        batch_y = batch_y.to(device, non_blocking=(device.type == "cuda"))

        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = head(model(batch_x))
        else:
            logits = head(model(batch_x))

        preds = logits.argmax(dim=1)
        all_true.append(batch_y.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.zeros((0,), dtype=np.int64)
    return classification_metrics(y_true, y_pred)


def _filter_unseen_eval_labels(
    train_labels: np.ndarray,
    flows: np.ndarray,
    labels: np.ndarray,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    train_classes = np.unique(np.asarray(train_labels).astype(str))
    labels_arr = np.asarray(labels)
    if labels_arr.size == 0:
        return np.asarray(flows), labels_arr, 0, []

    labels_str = labels_arr.astype(str)
    keep_mask = np.isin(labels_str, train_classes)
    dropped = int((~keep_mask).sum())
    if dropped == 0:
        return np.asarray(flows), labels_arr, 0, []

    unseen_classes = np.unique(labels_str[~keep_mask]).tolist()
    print(
        f"[warn] Dropping {dropped} {split_name} samples from "
        f"{len(unseen_classes)} labels unseen in train"
    )
    return np.asarray(flows)[keep_mask], labels_arr[keep_mask], dropped, unseen_classes


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
        replace_unseen_packet_threshold=max(0, args.replace_unseen_threshold),
    ).validated()

    loaded = load_embedder(settings=settings, pretrained=True)
    model = loaded.model
    ppi_transform = loaded.ppi_transform
    embedding_dim = loaded.output_dim

    device = torch.device(settings.device)

    splits = load_npz_splits(args.dataset_npz)

    val_flows, val_labels, val_dropped_unseen, val_unseen_classes = _filter_unseen_eval_labels(
        train_labels=splits.train_labels,
        flows=splits.val_flows,
        labels=splits.val_labels,
        split_name="validation",
    )
    test_flows, test_labels, test_dropped_unseen, test_unseen_classes = _filter_unseen_eval_labels(
        train_labels=splits.train_labels,
        flows=splits.test_flows,
        labels=splits.test_labels,
        split_name="test",
    )

    train_x = ppi_transform(np.asarray(splits.train_flows, dtype=np.float32)).astype(np.float32)
    val_x = ppi_transform(np.asarray(val_flows, dtype=np.float32)).astype(np.float32)
    test_x = ppi_transform(np.asarray(test_flows, dtype=np.float32)).astype(np.float32)

    encoder, train_y, (val_y, test_y) = encode_labels(
        splits.train_labels,
        val_labels,
        test_labels,
    )

    train_ds = FlowDataset(train_x, train_y)
    val_ds = FlowDataset(val_x, val_y)
    test_ds = FlowDataset(test_x, test_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(512, args.batch_size), shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=max(512, args.batch_size), shuffle=False, drop_last=False)

    head = nn.Sequential(
        nn.Dropout(args.dropout),
        nn.Linear(embedding_dim, len(encoder.classes_)),
    ).to(device)

    params = list(model.parameters()) + list(head.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * max(1, len(train_loader)))
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scaler = torch.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0
    ckpt_path = os.path.join(args.output_dir, "best_finetune.pt")

    global_step = 0
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        head.train()
        running_loss = 0.0
        num_seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device, non_blocking=(device.type == "cuda"))
            batch_y = batch_y.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)

            if args.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = head(model(batch_x))
                    loss = criterion(logits, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = head(model(batch_x))
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            scheduler.step()
            global_step += 1
            batch_size = batch_y.shape[0]
            running_loss += float(loss.item()) * batch_size
            num_seen += batch_size
            pbar.set_postfix(loss=f"{running_loss / max(1, num_seen):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = running_loss / max(1, num_seen)
        val_metrics = evaluate(model=model, head=head, loader=val_loader, device=device, amp=args.amp)
        val_acc = val_metrics["top1_acc"]

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['top1_acc']:.4f} "
            f"val_recall={val_metrics['macro_recall']:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "head_state": head.state_dict(),
                    "classes": encoder.classes_,
                    "feature_mode": settings.feature_mode,
                    "embedding_dim": embedding_dim,
                    "epoch": epoch,
                    "args": vars(args),
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"Early stopping: no validation improvement for {args.patience} epochs")
                break

    if not os.path.exists(ckpt_path):
        raise RuntimeError("Training finished without a valid checkpoint")

    best_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state"])
    head.load_state_dict(best_state["head_state"])

    test_metrics = evaluate(model=model, head=head, loader=test_loader, device=device, amp=args.amp)

    save_json(
        os.path.join(args.output_dir, "metrics.json"),
        {
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_metrics": test_metrics,
            "num_classes": int(len(encoder.classes_)),
            "dropped_unseen_eval_labels": {
                "val_samples": int(val_dropped_unseen),
                "test_samples": int(test_dropped_unseen),
                "val_label_count": int(len(val_unseen_classes)),
                "test_label_count": int(len(test_unseen_classes)),
                "val_labels": val_unseen_classes,
                "test_labels": test_unseen_classes,
            },
            "elapsed_sec": time.time() - started,
            "args": vars(args),
        },
    )

    np.save(os.path.join(args.output_dir, "label_classes.npy"), encoder.classes_, allow_pickle=True)

    print("Finetuning complete")
    print(f"Best checkpoint: {ckpt_path}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
