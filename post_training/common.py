from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DatasetSplits:
    train_flows: np.ndarray
    train_labels: np.ndarray
    val_flows: np.ndarray
    val_labels: np.ndarray
    test_flows: np.ndarray
    test_labels: np.ndarray


def _pick_key(payload: dict, candidates: list[str], required: bool = True):
    for key in candidates:
        if key in payload:
            return payload[key]
    if required:
        raise KeyError(f"Missing required keys. Expected one of: {candidates}")
    return None


def load_npz_splits(path: str) -> DatasetSplits:
    payload = np.load(path, allow_pickle=True)

    train_flows = _pick_key(payload, ["train_flows", "train_data", "x_train", "train_x"])  # type: ignore[arg-type]
    train_labels = _pick_key(payload, ["train_labels", "y_train", "train_y"])  # type: ignore[arg-type]

    val_flows = _pick_key(payload, ["val_flows", "val_data", "x_val", "val_x"], required=False)  # type: ignore[arg-type]
    val_labels = _pick_key(payload, ["val_labels", "y_val", "val_y"], required=False)  # type: ignore[arg-type]

    test_flows = _pick_key(payload, ["test_flows", "test_data", "x_test", "test_x"])  # type: ignore[arg-type]
    test_labels = _pick_key(payload, ["test_labels", "y_test", "test_y"])  # type: ignore[arg-type]

    if val_flows is None or val_labels is None:
        val_flows = np.zeros((0, 3, 30), dtype=np.float32)
        val_labels = np.zeros((0,), dtype=np.int64)

    return DatasetSplits(
        train_flows=np.asarray(train_flows),
        train_labels=np.asarray(train_labels),
        val_flows=np.asarray(val_flows),
        val_labels=np.asarray(val_labels),
        test_flows=np.asarray(test_flows),
        test_labels=np.asarray(test_labels),
    )


class FlowDataset(Dataset):
    def __init__(self, flows: np.ndarray, labels: np.ndarray):
        if len(flows) != len(labels):
            raise ValueError("flows and labels length mismatch")
        self.flows = torch.from_numpy(np.ascontiguousarray(flows, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.flows[idx], self.labels[idx]


def encode_labels(train_labels: np.ndarray, *others: np.ndarray) -> tuple[LabelEncoder, np.ndarray, list[np.ndarray]]:
    encoder = LabelEncoder()
    train_encoded = encoder.fit_transform(train_labels)
    encoded_others = []
    for labels in others:
        if len(labels) == 0:
            encoded_others.append(np.zeros((0,), dtype=np.int64))
        else:
            encoded_others.append(encoder.transform(labels))
    return encoder, train_encoded, encoded_others


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    top1_acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) > 0 else 0.0
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if len(y_true) > 0 else 0.0
    return {
        "top1_acc": top1_acc,
        "macro_recall": macro_recall,
        "weighted_f1": weighted_f1,
    }


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
