from __future__ import annotations

import argparse
import copy
import math
import os
import re
import time

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

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
from tc_transfer.finetune_utils.regularization import (
    LDIFSRegularization,
    SPRegularization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-parameter finetuning for downstream TC tasks"
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
    parser.add_argument("--replace-unseen-threshold", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="weighted_f1",
        choices=["top1_acc", "macro_recall", "weighted_f1"],
        help="Validation metric used for model selection and early stopping",
    )
    parser.add_argument(
        "--class-weighting",
        type=str,
        default="none",
        choices=["none", "inverse", "sqrt_inverse", "effective_num"],
        help="Class weighting strategy for cross-entropy loss",
    )
    parser.add_argument(
        "--class-weight-beta",
        type=float,
        default=0.9999,
        help="Beta for effective_num class weighting (recommended in [0.99, 0.9999])",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="random",
        choices=["random", "weighted"],
        help="Training sampler strategy",
    )
    parser.add_argument(
        "--sampler-power",
        type=float,
        default=0.5,
        help="Weighted sampler exponent: sample weight = class_count^(-power)",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm; <=0 disables clipping",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader worker processes"
    )
    parser.add_argument(
        "--pin-memory", action="store_true", help="Enable pin_memory for DataLoader"
    )
    parser.add_argument(
        "--embedder-finetuning",
        type=str,
        default="standard",
        choices=["standard", "layerwise_lr"],
        help="Embedder finetuning strategy",
    )
    parser.add_argument(
        "--layerwise-lr-mult",
        type=float,
        default=0.7,
        help="Layerwise LR multiplier used when embedder-finetuning=layerwise_lr",
    )
    parser.add_argument(
        "--embedder-batchnorm-eval-mode",
        action="store_true",
        help="Keep embedder BatchNorm in eval mode during training",
    )
    parser.add_argument(
        "--embedder-freeze-batchnorm-affine",
        action="store_true",
        help="Freeze BatchNorm affine parameters when using embedder-batchnorm-eval-mode",
    )
    parser.add_argument(
        "--embedder-dropout-eval-mode",
        action="store_true",
        help="Keep embedder Dropout in eval mode during training",
    )
    parser.add_argument(
        "--start-point-reg-alpha",
        type=float,
        default=0.0,
        help="L2SP regularization strength",
    )
    parser.add_argument(
        "--feature-space-reg-alpha",
        type=float,
        default=0.0,
        help="LDIFS regularization strength",
    )
    parser.add_argument(
        "--selection-start-epoch-frac",
        type=float,
        default=0.0,
        help="Start model selection after this epoch fraction (0.5 reproduces paper behavior)",
    )
    parser.add_argument(
        "--amp", action="store_true", help="Enable mixed precision on CUDA"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float
) -> LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
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


def _make_class_weights(
    train_labels: np.ndarray,
    num_classes: int,
    strategy: str,
    beta: float,
) -> np.ndarray | None:
    if strategy == "none":
        return None

    counts = np.bincount(train_labels.astype(np.int64), minlength=num_classes).astype(
        np.float64
    )
    counts = np.maximum(counts, 1.0)

    if strategy == "inverse":
        weights = 1.0 / counts
    elif strategy == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts)
    elif strategy == "effective_num":
        beta = float(min(max(beta, 0.0), 0.999999))
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise ValueError(f"Unsupported class weighting: {strategy}")

    # Keep average loss scale stable regardless of number of classes.
    weights = weights / max(weights.mean(), 1e-12)
    return weights.astype(np.float32)


def _build_train_sampler(
    train_labels: np.ndarray, strategy: str, power: float
) -> WeightedRandomSampler | None:
    if strategy == "random":
        return None

    if strategy != "weighted":
        raise ValueError(f"Unsupported sampler strategy: {strategy}")

    labels = train_labels.astype(np.int64)
    counts = np.bincount(labels).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    per_class_weight = np.power(counts, -float(power))
    sample_weights = per_class_weight[labels]
    sample_weights = sample_weights / max(sample_weights.mean(), 1e-12)

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype(np.float64)),
        num_samples=len(sample_weights),
        replacement=True,
    )


def _set_dropout_eval(module: nn.Module) -> None:
    if "Dropout" in module.__class__.__name__:
        module.eval()


def _set_batchnorm_eval(module: nn.Module) -> None:
    if "BatchNorm" in module.__class__.__name__:
        module.eval()


def _freeze_batchnorm_affine(model: nn.Module) -> None:
    for module in model.modules():
        if "BatchNorm" in module.__class__.__name__:
            for param in module.parameters():
                param.requires_grad = False


def _split_decay_groups(
    named_parameters: list[tuple[str, nn.Parameter]],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    dec_params: list[nn.Parameter] = []
    no_dec_params: list[nn.Parameter] = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        no_decay = param.ndim <= 1 or name.endswith(".bias") or "embedding" in name
        if no_decay:
            no_dec_params.append(param)
        else:
            dec_params.append(param)
    return dec_params, no_dec_params


def _setup_optimizer(
    model: nn.Module,
    head: nn.Module,
    lr: float,
    weight_decay: float,
    embedder_finetuning: str,
    layerwise_lr_mult: float,
) -> AdamW:
    param_groups: list[dict] = []

    def add_group(
        params: list[nn.Parameter], group_lr: float, group_weight_decay: float
    ) -> None:
        if params:
            param_groups.append(
                {"params": params, "lr": group_lr, "weight_decay": group_weight_decay}
            )

    # Head always uses the base learning rate.
    head_dec, head_no_dec = _split_decay_groups(list(head.named_parameters()))
    add_group(head_dec, lr, weight_decay)
    add_group(head_no_dec, lr, 0.0)

    if embedder_finetuning == "standard":
        embed_dec, embed_no_dec = _split_decay_groups(list(model.named_parameters()))
        add_group(embed_dec, lr, weight_decay)
        add_group(embed_no_dec, lr, 0.0)
        return AdamW(param_groups)

    if embedder_finetuning != "layerwise_lr":
        raise ValueError(f"Unsupported embedder_finetuning={embedder_finetuning}")

    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    block_regex = re.compile(r"backbone_model\.cnn_ppi\.(\d+)\.")
    block_indices = sorted(
        {
            int(match.group(1))
            for name, _ in named_params
            for match in [block_regex.search(name)]
            if match
        }
    )
    if not block_indices:
        # Fallback to standard if the architecture does not expose cnn block names.
        print(
            "[warn] layerwise_lr requested but no backbone_model.cnn_ppi.* blocks were detected; falling back to standard LR"
        )
        embed_dec, embed_no_dec = _split_decay_groups(named_params)
        add_group(embed_dec, lr, weight_decay)
        add_group(embed_no_dec, lr, 0.0)
        return AdamW(param_groups)

    max_idx = max(block_indices)
    num_blocks = len(block_indices)
    grouped: dict[tuple[float, bool], list[nn.Parameter]] = {}

    for name, param in named_params:
        lr_scale = 1.0
        block_match = block_regex.search(name)
        if block_match:
            idx = int(block_match.group(1))
            depth = max_idx - idx + 1
            lr_scale = layerwise_lr_mult**depth
        elif "packet_size_nn_embedding" in name or "packet_ipt_nn_embedding" in name:
            lr_scale = layerwise_lr_mult ** (num_blocks + 1)

        no_decay = param.ndim <= 1 or name.endswith(".bias") or "embedding" in name
        key = (lr_scale, no_decay)
        grouped.setdefault(key, []).append(param)

    for (lr_scale, no_decay), params in grouped.items():
        add_group(params, lr * lr_scale, 0.0 if no_decay else weight_decay)

    return AdamW(param_groups)


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

    val_flows, val_labels, val_dropped_unseen, val_unseen_classes = (
        _filter_unseen_eval_labels(
            train_labels=splits.train_labels,
            flows=splits.val_flows,
            labels=splits.val_labels,
            split_name="validation",
        )
    )
    test_flows, test_labels, test_dropped_unseen, test_unseen_classes = (
        _filter_unseen_eval_labels(
            train_labels=splits.train_labels,
            flows=splits.test_flows,
            labels=splits.test_labels,
            split_name="test",
        )
    )

    train_x = ppi_transform(np.asarray(splits.train_flows, dtype=np.float32)).astype(
        np.float32
    )
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

    if args.embedder_batchnorm_eval_mode and args.embedder_freeze_batchnorm_affine:
        _freeze_batchnorm_affine(model)

    num_workers = max(0, int(args.num_workers))
    use_pin_memory = bool(args.pin_memory and device.type == "cuda")
    train_sampler = _build_train_sampler(
        train_y, strategy=args.sampler, power=args.sampler_power
    )

    loader_common_kwargs: dict = {
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
    }
    if num_workers > 0:
        loader_common_kwargs["persistent_workers"] = True
        loader_common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=False,
        **loader_common_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(512, args.batch_size),
        shuffle=False,
        drop_last=False,
        **loader_common_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(512, args.batch_size),
        shuffle=False,
        drop_last=False,
        **loader_common_kwargs,
    )

    head = nn.Sequential(
        nn.Dropout(args.dropout),
        nn.Linear(embedding_dim, len(encoder.classes_)),
    ).to(device)

    params = list(model.parameters()) + list(head.parameters())
    optimizer = _setup_optimizer(
        model=model,
        head=head,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embedder_finetuning=args.embedder_finetuning,
        layerwise_lr_mult=args.layerwise_lr_mult,
    )
    total_steps = max(1, args.epochs * max(1, len(train_loader)))
    scheduler = build_scheduler(
        optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio
    )

    class_weights_np = _make_class_weights(
        train_labels=train_y,
        num_classes=len(encoder.classes_),
        strategy=args.class_weighting,
        beta=args.class_weight_beta,
    )
    class_weights_t = (
        None
        if class_weights_np is None
        else torch.from_numpy(class_weights_np).to(device)
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_t, label_smoothing=args.label_smoothing
    )

    reference_model: nn.Module | None = None
    start_point_reg_fn: SPRegularization | None = None
    feature_space_reg_fn: LDIFSRegularization | None = None
    if args.start_point_reg_alpha > 0 or args.feature_space_reg_alpha > 0:
        reference_model = copy.deepcopy(model).to(device)
        reference_model.eval()
        for ref_param in reference_model.parameters():
            ref_param.requires_grad = False
        if args.start_point_reg_alpha > 0:
            start_point_reg_fn = SPRegularization(
                source_model=reference_model, target_model=model
            )
        if args.feature_space_reg_alpha > 0:
            feature_space_reg_fn = LDIFSRegularization(
                source_model=reference_model, target_model=model
            )

    scaler = torch.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_metric = -1.0
    best_epoch = -1
    no_improve = 0
    ckpt_path = os.path.join(args.output_dir, "best_finetune.pt")
    history: list[dict[str, float]] = []
    selection_start_epoch = max(
        0,
        min(
            args.epochs - 1,
            int(args.epochs * max(0.0, min(args.selection_start_epoch_frac, 1.0))),
        ),
    )

    global_step = 0
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        head.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_sp_loss = 0.0
        running_fs_loss = 0.0
        num_seen = 0

        if args.embedder_dropout_eval_mode:
            model.apply(_set_dropout_eval)
        if args.embedder_batchnorm_eval_mode:
            model.apply(_set_batchnorm_eval)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device, non_blocking=(device.type == "cuda"))
            batch_y = batch_y.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)

            if args.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = head(model(batch_x))
                    ce_loss = criterion(logits, batch_y)
                    fs_reg_loss = torch.zeros((), device=device)
                    sp_reg_loss = torch.zeros((), device=device)
                    if feature_space_reg_fn is not None:
                        fs_reg_loss = (
                            args.feature_space_reg_alpha * feature_space_reg_fn(batch_x)
                        )
                    if start_point_reg_fn is not None:
                        sp_reg_loss = args.start_point_reg_alpha * start_point_reg_fn()
                    loss = ce_loss + fs_reg_loss + sp_reg_loss
                scaler.scale(loss).backward()

                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                logits = head(model(batch_x))
                ce_loss = criterion(logits, batch_y)
                fs_reg_loss = torch.zeros((), device=device)
                sp_reg_loss = torch.zeros((), device=device)
                if feature_space_reg_fn is not None:
                    fs_reg_loss = args.feature_space_reg_alpha * feature_space_reg_fn(
                        batch_x
                    )
                if start_point_reg_fn is not None:
                    sp_reg_loss = args.start_point_reg_alpha * start_point_reg_fn()
                loss = ce_loss + fs_reg_loss + sp_reg_loss
                loss.backward()

                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip_norm)

                optimizer.step()

            scheduler.step()
            global_step += 1
            batch_size = batch_y.shape[0]
            running_loss += float(loss.item()) * batch_size
            running_ce_loss += float(ce_loss.item()) * batch_size
            running_sp_loss += float(sp_reg_loss.item()) * batch_size
            running_fs_loss += float(fs_reg_loss.item()) * batch_size
            num_seen += batch_size
            pbar.set_postfix(
                loss=f"{running_loss / max(1, num_seen):.4f}",
                ce=f"{running_ce_loss / max(1, num_seen):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        train_loss = running_loss / max(1, num_seen)
        val_metrics = evaluate(
            model=model, head=head, loader=val_loader, device=device, amp=args.amp
        )
        if args.selection_metric not in val_metrics:
            raise ValueError(
                f"selection_metric={args.selection_metric} is missing in validation metrics: {list(val_metrics.keys())}"
            )
        selected_val_metric = float(val_metrics[args.selection_metric])
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "val_top1_acc": float(val_metrics["top1_acc"]),
                "val_macro_recall": float(val_metrics["macro_recall"]),
                "val_weighted_f1": float(val_metrics["weighted_f1"]),
                "val_selection_metric": selected_val_metric,
            }
        )

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} "
            f"(ce={running_ce_loss / max(1, num_seen):.4f}, "
            f"sp={running_sp_loss / max(1, num_seen):.4f}, "
            f"fs={running_fs_loss / max(1, num_seen):.4f}) "
            f"val_acc={val_metrics['top1_acc']:.4f} "
            f"val_recall={val_metrics['macro_recall']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f} "
            f"select({args.selection_metric})={selected_val_metric:.4f}"
        )

        if epoch >= selection_start_epoch and selected_val_metric > best_val_metric:
            best_val_metric = selected_val_metric
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
                    "selection_metric": args.selection_metric,
                    "best_val_metric": best_val_metric,
                    "selection_start_epoch": selection_start_epoch,
                    "args": vars(args),
                },
                ckpt_path,
            )
        elif epoch >= selection_start_epoch:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(
                    f"Early stopping: no validation improvement for {args.patience} epochs"
                )
                break

    if not os.path.exists(ckpt_path):
        raise RuntimeError("Training finished without a valid checkpoint")

    best_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state"])
    head.load_state_dict(best_state["head_state"])

    test_metrics = evaluate(
        model=model, head=head, loader=test_loader, device=device, amp=args.amp
    )

    save_json(
        os.path.join(args.output_dir, "metrics.json"),
        {
            "best_epoch": best_epoch,
            "selection_metric": args.selection_metric,
            "best_val_metric": best_val_metric,
            "test_metrics": test_metrics,
            "num_classes": int(len(encoder.classes_)),
            "class_weighting": args.class_weighting,
            "class_weight_beta": args.class_weight_beta,
            "sampler": args.sampler,
            "sampler_power": args.sampler_power,
            "embedder_finetuning": args.embedder_finetuning,
            "layerwise_lr_mult": args.layerwise_lr_mult,
            "embedder_batchnorm_eval_mode": bool(args.embedder_batchnorm_eval_mode),
            "embedder_freeze_batchnorm_affine": bool(
                args.embedder_freeze_batchnorm_affine
            ),
            "embedder_dropout_eval_mode": bool(args.embedder_dropout_eval_mode),
            "start_point_reg_alpha": float(args.start_point_reg_alpha),
            "feature_space_reg_alpha": float(args.feature_space_reg_alpha),
            "selection_start_epoch": int(selection_start_epoch),
            "history": history,
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

    np.save(
        os.path.join(args.output_dir, "label_classes.npy"),
        encoder.classes_,
        allow_pickle=True,
    )

    print("Finetuning complete")
    print(f"Best checkpoint: {ckpt_path}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
