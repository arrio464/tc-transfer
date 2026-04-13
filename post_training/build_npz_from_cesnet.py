from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable


@dataclass
class ConversionStats:
    input_rows: int
    valid_rows: int
    kept_rows: int
    dropped_missing: int
    dropped_parse_error: int
    dropped_low_freq: int
    dropped_val_unseen: int
    dropped_test_unseen: int
    num_classes: int
    train_size: int
    val_size: int
    test_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CESNET QUIC CSV/Parquet to NPZ expected by post_training scripts"
    )
    parser.add_argument("--input-path", required=True, help="Source CSV/Parquet path")
    parser.add_argument("--output-npz", required=True, help="Output NPZ path")

    parser.add_argument("--ppi-column", default="PPI", help="Column containing packet sequence triplet")
    parser.add_argument("--label-column", default="QUIC_SNI", help="Target label column")

    parser.add_argument("--max-packets", type=int, default=30, help="Truncate/pad packet sequence length")
    parser.add_argument("--min-class-count", type=int, default=3, help="Drop labels with fewer samples")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap after filtering; 0 means no cap")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument(
        "--direction-fix",
        choices=["none", "zero_to_minus_one", "sign"],
        default="none",
        help="Optional normalization of packet directions",
    )
    parser.add_argument("--size-clip-min", type=float, default=0.0)
    parser.add_argument("--size-clip-max", type=float, default=1500.0)
    parser.add_argument("--ipt-clip-min", type=float, default=0.0)
    parser.add_argument("--ipt-clip-max", type=float, default=60000.0)
    parser.add_argument("--nrows", type=int, default=0, help="Optional preview limit for CSV")
    return parser.parse_args()


def _read_table(input_path: str, ppi_column: str, label_column: str, nrows: int) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for CSV/Parquet conversion. Install pandas and pyarrow.") from exc

    ext = os.path.splitext(input_path)[1].lower()
    usecols = [ppi_column, label_column]

    if ext == ".csv":
        return pd.read_csv(input_path, usecols=usecols, nrows=(None if nrows <= 0 else nrows))

    if ext in {".parquet", ".pq"} or os.path.isdir(input_path):
        # Nested PPI arrays require pyarrow for reliable decoding.
        table = pd.read_parquet(input_path, columns=usecols, engine="pyarrow")  # type: ignore[arg-type]
        if nrows > 0:
            return table.iloc[:nrows].copy()
        return table

    raise ValueError("Unsupported input format. Use CSV or Parquet.")


def _parse_ppi(raw: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = raw
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = ast.literal_eval(value)

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("PPI must be [ipt, dir, size]")

    ipt = np.asarray(value[0], dtype=np.float32)
    direction = np.asarray(value[1], dtype=np.float32)
    size = np.asarray(value[2], dtype=np.float32)

    if not (len(ipt) == len(direction) == len(size)):
        raise ValueError("PPI channels must have same length")

    return ipt, direction, size


def _to_fixed_ppi(
    ipt: np.ndarray,
    direction: np.ndarray,
    size: np.ndarray,
    max_packets: int,
    direction_fix: str,
    size_clip_min: float,
    size_clip_max: float,
    ipt_clip_min: float,
    ipt_clip_max: float,
) -> np.ndarray:
    ipt = np.clip(ipt, ipt_clip_min, ipt_clip_max)
    size = np.clip(size, size_clip_min, size_clip_max)

    if direction_fix == "zero_to_minus_one":
        direction = np.where(direction == 0, -1, direction)
    elif direction_fix == "sign":
        direction = np.sign(direction)

    seq_len = min(max_packets, len(ipt))
    out = np.zeros((3, max_packets), dtype=np.float32)
    out[0, :seq_len] = ipt[:seq_len]
    out[1, :seq_len] = direction[:seq_len]
    out[2, :seq_len] = size[:seq_len]
    return out


def _stratified_subsample(X: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    if max_samples <= 0 or len(X) <= max_samples:
        return X, y

    try:
        idx, _ = train_test_split(
            np.arange(len(X)),
            train_size=max_samples,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_samples, replace=False)

    return X[idx], y[idx]


def _split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Invalid split ratios")

    if val_ratio == 0 and test_ratio == 0:
        raise ValueError("At least one of val_ratio or test_ratio must be > 0")

    stratify = y if len(np.unique(y)) > 1 else None

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        train_size=train_ratio,
        random_state=seed,
        stratify=stratify,
    )

    if val_ratio == 0:
        return X_train, y_train, np.zeros((0, 3, X.shape[2]), dtype=np.float32), np.zeros((0,), dtype=y.dtype), X_tmp, y_tmp
    if test_ratio == 0:
        return X_train, y_train, X_tmp, y_tmp, np.zeros((0, 3, X.shape[2]), dtype=np.float32), np.zeros((0,), dtype=y.dtype)

    val_frac_in_tmp = val_ratio / (val_ratio + test_ratio)
    stratify_tmp = y_tmp if len(np.unique(y_tmp)) > 1 else None

    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp,
            y_tmp,
            train_size=val_frac_in_tmp,
            random_state=seed + 1,
            stratify=stratify_tmp,
        )
    except ValueError:
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp,
            y_tmp,
            train_size=val_frac_in_tmp,
            random_state=seed + 1,
            stratify=None,
        )

    return X_train, y_train, X_val, y_val, X_test, y_test


def _drop_unseen_eval_labels(
    train_labels: np.ndarray,
    eval_flows: np.ndarray,
    eval_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    train_classes = np.unique(np.asarray(train_labels).astype(str))
    eval_labels_arr = np.asarray(eval_labels).astype(str)
    if eval_labels_arr.size == 0:
        return np.asarray(eval_flows), eval_labels_arr, 0

    keep_mask = np.isin(eval_labels_arr, train_classes)
    dropped = int((~keep_mask).sum())
    if dropped == 0:
        return np.asarray(eval_flows), eval_labels_arr, 0
    return np.asarray(eval_flows)[keep_mask], eval_labels_arr[keep_mask], dropped


def main() -> None:
    args = parse_args()

    table = _read_table(args.input_path, args.ppi_column, args.label_column, args.nrows)
    input_rows = len(table)

    missing_mask = table[args.ppi_column].isna() | table[args.label_column].isna() | (table[args.label_column].astype(str).str.len() == 0)
    dropped_missing = int(missing_mask.sum())
    table = table[~missing_mask].copy()

    flows: list[np.ndarray] = []
    labels: list[str] = []
    dropped_parse_error = 0

    for raw_ppi, raw_label in tqdm(
        zip(table[args.ppi_column].tolist(), table[args.label_column].tolist()),
        total=len(table),
        desc="Parsing PPI",
    ):
        try:
            ipt, direction, size = _parse_ppi(raw_ppi)
            fixed = _to_fixed_ppi(
                ipt=ipt,
                direction=direction,
                size=size,
                max_packets=args.max_packets,
                direction_fix=args.direction_fix,
                size_clip_min=args.size_clip_min,
                size_clip_max=args.size_clip_max,
                ipt_clip_min=args.ipt_clip_min,
                ipt_clip_max=args.ipt_clip_max,
            )
        except Exception:
            dropped_parse_error += 1
            continue

        flows.append(fixed)
        labels.append(str(raw_label))

    if not flows:
        raise RuntimeError("No valid samples after parsing")

    X = np.stack(flows).astype(np.float32)
    y = np.asarray(labels)
    valid_rows = len(y)

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for class-frequency filtering. Install pandas.") from exc

    counts = pd.Series(y).value_counts()
    keep_labels = counts[counts >= args.min_class_count].index
    keep_mask = np.isin(y, keep_labels)
    dropped_low_freq = int((~keep_mask).sum())

    X = X[keep_mask]
    y = y[keep_mask]

    if len(np.unique(y)) < 2:
        raise RuntimeError("Need at least two classes after filtering")

    X, y = _stratified_subsample(X, y, args.max_samples, args.seed)
    kept_rows = len(y)

    try:
        train_flows, train_labels, val_flows, val_labels, test_flows, test_labels = _split_dataset(
            X=X,
            y=y,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    except ValueError:
        # Fallback without stratification in degenerate class-count scenarios.
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(X))
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

        n_train = int(len(X) * args.train_ratio)
        n_val = int(len(X) * args.val_ratio)
        train_flows = X[:n_train]
        train_labels = y[:n_train]
        val_flows = X[n_train:n_train + n_val]
        val_labels = y[n_train:n_train + n_val]
        test_flows = X[n_train + n_val:]
        test_labels = y[n_train + n_val:]

    val_flows, val_labels, dropped_val_unseen = _drop_unseen_eval_labels(
        train_labels=train_labels,
        eval_flows=val_flows,
        eval_labels=val_labels,
    )
    test_flows, test_labels, dropped_test_unseen = _drop_unseen_eval_labels(
        train_labels=train_labels,
        eval_flows=test_flows,
        eval_labels=test_labels,
    )

    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)

    np.savez_compressed(
        args.output_npz,
        train_flows=train_flows,
        train_labels=train_labels,
        val_flows=val_flows,
        val_labels=val_labels,
        test_flows=test_flows,
        test_labels=test_labels,
    )

    stats = ConversionStats(
        input_rows=input_rows,
        valid_rows=valid_rows,
        kept_rows=kept_rows,
        dropped_missing=dropped_missing,
        dropped_parse_error=dropped_parse_error,
        dropped_low_freq=dropped_low_freq,
        dropped_val_unseen=dropped_val_unseen,
        dropped_test_unseen=dropped_test_unseen,
        num_classes=int(len(np.unique(y))),
        train_size=int(len(train_flows)),
        val_size=int(len(val_flows)),
        test_size=int(len(test_flows)),
    )

    summary_path = os.path.splitext(args.output_npz)[0] + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(asdict(stats), fp, indent=2)

    print("Conversion finished")
    print(f"Output NPZ: {args.output_npz}")
    print(f"Summary: {summary_path}")
    print(json.dumps(asdict(stats), indent=2))


if __name__ == "__main__":
    main()
