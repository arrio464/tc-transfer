from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import requests

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "pandas is required. Install it with: pip install pandas"
    ) from exc


SAMPLES_CSV = "G:\\cesnet-quic22\\w44_washed_sample.csv"
# SAMPLES_CSV = "G:\\cesnet-quic22\\w46_washed_sample.csv"
API_BASE_URL = "172.31.172.100:8080"


@dataclass(frozen=True)
class Sample:
    flow: list[list[float]]
    label: str


@dataclass
class RequestResult:
    ok: bool
    status_code: int | None
    latency_ms: float
    endpoint_elapsed_ms: float | None
    batch_size: int
    predictions: list[str | None]
    labels: list[str]
    topk_hits: int
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concurrent benchmark + correctness evaluation for inference backend /predict/knn"
        )
    )
    parser.add_argument(
        "--csv", default=SAMPLES_CSV, help="Path to CSV containing PPI and labels"
    )
    parser.add_argument(
        "--base-url",
        default=API_BASE_URL,
        help="Backend URL, e.g. http://127.0.0.1:8080",
    )
    parser.add_argument(
        "--ppi-column", default="PPI", help="Column containing [ipt, dir, size]"
    )
    parser.add_argument(
        "--label-column", default="CATEGORY", help="Ground-truth label column"
    )
    parser.add_argument("--top-k", type=int, default=5, help="top_k for /predict/knn")
    parser.add_argument(
        "--max-packets", type=int, default=30, help="Expected packet length"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of rows to evaluate, 0 = all",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Flows per request")
    parser.add_argument(
        "--workers", type=int, default=16, help="Thread count for concurrent requests"
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=5,
        help="Warm-up requests before measuring",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="HTTP request timeout in seconds"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retry times for transient request failures",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle selected samples before splitting into batches",
    )
    parser.add_argument(
        "--index-path",
        default="",
        help="Optional: index.faiss path to load before benchmark",
    )
    parser.add_argument(
        "--labels-path",
        default="",
        help="Optional: labels.npy path to load before benchmark",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional: write report JSON to file",
    )
    return parser.parse_args()


def normalize_base_url(base_url: str) -> str:
    candidate = base_url.strip().rstrip("/")
    if not candidate.startswith(("http://", "https://")):
        candidate = f"http://{candidate}"
    return candidate


def parse_ppi(raw: Any, max_packets: int) -> list[list[float]]:
    value = raw
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = ast.literal_eval(value)

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("PPI must be [ipt, dir, size]")

    ipt = list(value[0])
    direction = list(value[1])
    size = list(value[2])
    if not (len(ipt) == len(direction) == len(size)):
        raise ValueError("PPI channels must have the same length")

    n = min(max_packets, len(ipt))
    out = [[0.0] * max_packets for _ in range(3)]
    for i in range(n):
        out[0][i] = float(ipt[i])
        out[1][i] = float(direction[i])
        out[2][i] = float(size[i])
    return out


def read_samples(
    csv_path: str,
    ppi_column: str,
    label_column: str,
    max_packets: int,
    sample_size: int,
    seed: int,
    shuffle: bool,
) -> list[Sample]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, usecols=[ppi_column, label_column])
    df = df.dropna(subset=[ppi_column, label_column])

    rows = df.to_dict(orient="records")
    rng = random.Random(seed)
    if sample_size > 0 and sample_size < len(rows):
        rows = rng.sample(rows, sample_size)
    if shuffle:
        rng.shuffle(rows)

    samples: list[Sample] = []
    dropped = 0
    for row in rows:
        try:
            flow = parse_ppi(row[ppi_column], max_packets=max_packets)
            label = str(row[label_column])
            samples.append(Sample(flow=flow, label=label))
        except Exception:
            dropped += 1

    if not samples:
        raise RuntimeError("No valid samples available after parsing.")

    print(f"Loaded {len(samples)} valid samples from {csv_path} (dropped {dropped}).")
    return samples


def split_batches(samples: list[Sample], batch_size: int) -> list[list[Sample]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]


def create_session(timeout: float) -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=256, pool_maxsize=256, max_retries=0
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.request = _with_default_timeout(session.request, timeout)
    return session


def _with_default_timeout(request_fn, timeout: float):
    def wrapped(method: str, url: str, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout
        return request_fn(method, url, **kwargs)

    return wrapped


def health_check(session: requests.Session, base_url: str) -> dict[str, Any]:
    response = session.get(f"{base_url}/health")
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"Health check failed: {payload}")
    return payload


def maybe_load_index(
    session: requests.Session, base_url: str, index_path: str, labels_path: str
) -> dict[str, Any] | None:
    if not index_path and not labels_path:
        return None
    if not index_path or not labels_path:
        raise ValueError(
            "Both --index-path and --labels-path are required when loading index"
        )

    payload = {"index_path": index_path, "labels_path": labels_path}
    response = session.post(f"{base_url}/knn/index/load", json=payload)
    response.raise_for_status()
    return response.json()


def compute_topk_hit_rate(
    neighbors: list[list[dict[str, Any]]], labels: list[str]
) -> int:
    hits = 0
    for idx, neigh in enumerate(neighbors):
        gt = labels[idx]
        candidate_labels = [
            str(item.get("label")) for item in neigh if item.get("label") is not None
        ]
        if gt in candidate_labels:
            hits += 1
    return hits


def post_predict(
    session: requests.Session,
    base_url: str,
    batch: list[Sample],
    top_k: int,
    retries: int,
) -> RequestResult:
    flows = [s.flow for s in batch]
    labels = [s.label for s in batch]
    payload = {"flows": flows, "top_k": top_k}

    last_error: str | None = None
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            response = session.post(f"{base_url}/predict/knn", json=payload)
            latency_ms = (time.perf_counter() - start) * 1000.0
            status_code = response.status_code
            if status_code != 200:
                text = response.text[:400]
                last_error = f"HTTP {status_code}: {text}"
                if attempt < retries and status_code >= 500:
                    continue
                return RequestResult(
                    ok=False,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    endpoint_elapsed_ms=None,
                    batch_size=len(batch),
                    predictions=[],
                    labels=labels,
                    topk_hits=0,
                    error=last_error,
                )

            body = response.json()
            predictions = [
                None if p is None else str(p) for p in body.get("predictions", [])
            ]
            neighbors = body.get("neighbors", [])
            endpoint_elapsed_ms = body.get("elapsed_ms")
            topk_hits = compute_topk_hit_rate(neighbors=neighbors, labels=labels)

            if len(predictions) != len(labels):
                return RequestResult(
                    ok=False,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    endpoint_elapsed_ms=None,
                    batch_size=len(batch),
                    predictions=predictions,
                    labels=labels,
                    topk_hits=topk_hits,
                    error=(
                        "Prediction count mismatch: "
                        f"expected={len(labels)} got={len(predictions)}"
                    ),
                )

            return RequestResult(
                ok=True,
                status_code=status_code,
                latency_ms=latency_ms,
                endpoint_elapsed_ms=(
                    None if endpoint_elapsed_ms is None else float(endpoint_elapsed_ms)
                ),
                batch_size=len(batch),
                predictions=predictions,
                labels=labels,
                topk_hits=topk_hits,
                error=None,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            last_error = str(exc)
            if attempt >= retries:
                return RequestResult(
                    ok=False,
                    status_code=None,
                    latency_ms=latency_ms,
                    endpoint_elapsed_ms=None,
                    batch_size=len(batch),
                    predictions=[],
                    labels=labels,
                    topk_hits=0,
                    error=last_error,
                )

    return RequestResult(
        ok=False,
        status_code=None,
        latency_ms=0.0,
        endpoint_elapsed_ms=None,
        batch_size=len(batch),
        predictions=[],
        labels=labels,
        topk_hits=0,
        error=last_error,
    )


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    rank = (len(sorted_values) - 1) * (p / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_results(
    results: list[RequestResult], total_wall_s: float
) -> dict[str, Any]:
    total_requests = len(results)
    success_results = [r for r in results if r.ok]
    failed_results = [r for r in results if not r.ok]

    total_samples = sum(r.batch_size for r in results)
    success_samples = sum(r.batch_size for r in success_results)

    top1_hits = 0
    topk_hits = 0
    for r in success_results:
        topk_hits += r.topk_hits
        top1_hits += sum(1 for pred, gt in zip(r.predictions, r.labels) if pred == gt)

    request_latencies = sorted(r.latency_ms for r in results)
    endpoint_latencies = sorted(
        r.endpoint_elapsed_ms
        for r in success_results
        if r.endpoint_elapsed_ms is not None
    )

    req_per_sec = total_requests / total_wall_s if total_wall_s > 0 else float("nan")
    sample_per_sec = total_samples / total_wall_s if total_wall_s > 0 else float("nan")

    report: dict[str, Any] = {
        "requests": {
            "total": total_requests,
            "success": len(success_results),
            "failed": len(failed_results),
            "success_rate": (
                len(success_results) / total_requests if total_requests else 0.0
            ),
            "throughput_req_s": req_per_sec,
        },
        "samples": {
            "total": total_samples,
            "successful": success_samples,
            "throughput_sample_s": sample_per_sec,
        },
        "accuracy": {
            "top1_acc": (top1_hits / success_samples if success_samples else 0.0),
            "topk_hit_rate": (topk_hits / success_samples if success_samples else 0.0),
        },
        "latency_ms": {
            "request": {
                "min": request_latencies[0] if request_latencies else float("nan"),
                "mean": statistics.fmean(request_latencies)
                if request_latencies
                else float("nan"),
                "p50": percentile(request_latencies, 50),
                "p90": percentile(request_latencies, 90),
                "p95": percentile(request_latencies, 95),
                "p99": percentile(request_latencies, 99),
                "max": request_latencies[-1] if request_latencies else float("nan"),
            },
            "endpoint_reported": {
                "mean": statistics.fmean(endpoint_latencies)
                if endpoint_latencies
                else float("nan"),
                "p50": percentile(endpoint_latencies, 50),
                "p90": percentile(endpoint_latencies, 90),
                "p95": percentile(endpoint_latencies, 95),
                "p99": percentile(endpoint_latencies, 99),
            },
        },
        "errors": [
            {
                "status_code": r.status_code,
                "batch_size": r.batch_size,
                "error": r.error,
            }
            for r in failed_results[:20]
        ],
        "wall_time_sec": total_wall_s,
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    req = report["requests"]
    samples = report["samples"]
    acc = report["accuracy"]
    lat = report["latency_ms"]

    print("\n===== Load Test Report =====")
    print(
        "Requests: "
        f"total={req['total']} success={req['success']} failed={req['failed']} "
        f"success_rate={req['success_rate'] * 100:.2f}% throughput={req['throughput_req_s']:.2f} req/s"
    )
    print(
        "Samples: "
        f"total={samples['total']} successful={samples['successful']} "
        f"throughput={samples['throughput_sample_s']:.2f} samples/s"
    )
    print(
        "Accuracy: "
        f"top1={acc['top1_acc'] * 100:.2f}% topK_hit={acc['topk_hit_rate'] * 100:.2f}%"
    )
    print(
        "Request latency(ms): "
        f"mean={lat['request']['mean']:.2f} p50={lat['request']['p50']:.2f} "
        f"p90={lat['request']['p90']:.2f} p95={lat['request']['p95']:.2f} "
        f"p99={lat['request']['p99']:.2f}"
    )
    print(
        "Endpoint elapsed(ms): "
        f"mean={lat['endpoint_reported']['mean']:.2f} p50={lat['endpoint_reported']['p50']:.2f} "
        f"p90={lat['endpoint_reported']['p90']:.2f} p95={lat['endpoint_reported']['p95']:.2f} "
        f"p99={lat['endpoint_reported']['p99']:.2f}"
    )

    if report["errors"]:
        print("Top errors:")
        for idx, item in enumerate(report["errors"], start=1):
            print(
                f"  {idx}. status={item['status_code']} batch_size={item['batch_size']} error={item['error']}"
            )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    base_url = normalize_base_url(args.base_url)
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    samples = read_samples(
        csv_path=args.csv,
        ppi_column=args.ppi_column,
        label_column=args.label_column,
        max_packets=args.max_packets,
        sample_size=args.sample_size,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    batches = split_batches(samples=samples, batch_size=args.batch_size)
    print(
        f"Prepared {len(samples)} samples into {len(batches)} requests "
        f"(batch_size={args.batch_size}, workers={args.workers})."
    )

    # Keep one session per worker thread for connection reuse.
    thread_local = threading.local()

    def get_session() -> requests.Session:
        session = getattr(thread_local, "session", None)
        if session is None:
            session = create_session(timeout=args.timeout)
            thread_local.session = session
        return session

    bootstrap_session = create_session(timeout=args.timeout)
    health = health_check(bootstrap_session, base_url=base_url)
    print(f"Health: {health}")

    load_result = maybe_load_index(
        bootstrap_session,
        base_url=base_url,
        index_path=args.index_path,
        labels_path=args.labels_path,
    )
    if load_result is not None:
        print(f"Index loaded: {load_result}")

    if args.warmup_requests > 0:
        print(f"Running warm-up requests: {args.warmup_requests}")
        warmup_batch = batches[0]
        for _ in range(args.warmup_requests):
            _ = post_predict(
                session=bootstrap_session,
                base_url=base_url,
                batch=warmup_batch,
                top_k=args.top_k,
                retries=0,
            )

    print("Starting concurrent benchmark...")
    started_at = time.perf_counter()
    results: list[RequestResult] = []

    def run_batch(batch: list[Sample]) -> RequestResult:
        session = get_session()
        return post_predict(
            session=session,
            base_url=base_url,
            batch=batch,
            top_k=args.top_k,
            retries=args.retries,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(run_batch, batch): idx for idx, batch in enumerate(batches)
        }
        for done_count, future in enumerate(as_completed(future_to_idx), start=1):
            result = future.result()
            results.append(result)
            if done_count % max(1, len(batches) // 10) == 0 or done_count == len(
                batches
            ):
                print(f"Progress: {done_count}/{len(batches)} requests completed")

    total_wall_s = time.perf_counter() - started_at
    report = summarize_results(results=results, total_wall_s=total_wall_s)
    print_report(report)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        print(f"Saved JSON report to: {args.output_json}")


if __name__ == "__main__":
    main()
"""
80♥   G:\tc-transfer  on   main ±3 ?18                                                exit  0xc000013a  󰌠 [cuda128]
❯ python -m post_training.knn_transfer --dataset-npz cesnet-quic22-w47-washed.npz --output-dir cesnet-quic22-w47-washed-
knn --feature-mode backbone_gem --finetune-checkpoint cesnet-quic22-w47-washed/best_finetune.pt --vote majority --auto-t
une-k --k-candidates 1,3,5,9 --selection-metric top1_acc --metric cosine --amp --clear-cuda-cache-between-phases --faiss
-gpu
Loaded finetune checkpoint: cesnet-quic22-w47-washed/best_finetune.pt
Embedding train: 100%|█████████████████████████████████████████████████████████████████| 66/66 [00:06<00:00,  9.59it/s]
Embedding val: 100%|█████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00,  9.99it/s]
Embedding test: 100%|████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 10.13it/s]
kNN transfer complete
Selected K: 9
Test metrics: {'top1_acc': 0.9543731908925428, 'macro_recall': 0.9544229399838668, 'weighted_f1': 0.9544167690526761}
Index artifacts saved in: cesnet-quic22-w47-washed-knn\knn_index

% curl -X POST 127.0.0.1:8080/knn/index/load -H "Content-Type: application/json" -d '{"index_path": "cesnet-quic22-w47-washed-knn/knn_index/index.faiss", "labels_path": "cesnet-quic22-w47-washed-knn/knn_index/labels.npy"}'
{"message":"Index loaded","index_size":268080,"index_dim":448,"metric":"cosine"}

> D:/Files/scoop/persist/miniforge/cuda128/python.exe g:/tc-transfer/test.py
Loaded 5000 valid samples from G:\cesnet-quic22\w44_washed_sample.csv (dropped 0).
Prepared 5000 samples into 40 requests (batch_size=128, workers=16).
Health: {'status': 'ok', 'details': {'device': 'cpu', 'feature_mode': 'backbone_gem', 'embedding_dim': 448, 'knn_ready': True, 'knn_size': 268080, 'knn_metric': 'cosine'}}
Running warm-up requests: 5
Starting concurrent benchmark...
Progress: 4/40 requests completed
Progress: 8/40 requests completed
Progress: 12/40 requests completed
Progress: 16/40 requests completed
Progress: 20/40 requests completed
Progress: 24/40 requests completed
Progress: 28/40 requests completed
Progress: 32/40 requests completed
Progress: 36/40 requests completed
Progress: 40/40 requests completed

===== Load Test Report =====
Requests: total=40 success=40 failed=0 success_rate=100.00% throughput=1.47 req/s
Samples: total=5000 successful=5000 throughput=183.66 samples/s
Accuracy: top1=82.38% topK_hit=89.36%
Request latency(ms): mean=9137.34 p50=10592.90 p90=11206.41 p95=11353.96 p99=11660.69
Endpoint elapsed(ms): mean=9095.87 p50=10580.47 p90=11178.72 p95=11338.41 p99=11588.15

80♥   G:\tc-transfer  on   main ±3 ?18                                                       took 35s   󰌠 [cuda128]
❯ python -m post_training.knn_transfer --dataset-npz cesnet-quic22-w45-w46-w47-washed.npz --output-dir cesnet-quic22-w45-w46-w47-washed-knn --feature-mode backbone_gem --finetune-checkpoint cesnet-quic22-w45-w46-w47-washed/best_finetune.pt --vote majority --auto-tune-k --k-candidates 1,3,5,9 --selection-metric top1_acc --metric cosine --amp --clear-cuda-cach
e-between-phases --faiss-gpu
Loaded finetune checkpoint: cesnet-quic22-w45-w46-w47-washed/best_finetune.pt
Embedding train: 100%|███████████████████████████████████████████████████████████████| 257/257 [00:27<00:00,  9.21it/s]
Embedding val: 100%|███████████████████████████████████████████████████████████████████| 33/33 [00:03<00:00,  9.62it/s]
Embedding test: 100%|██████████████████████████████████████████████████████████████████| 33/33 [00:03<00:00,  8.56it/s]
kNN transfer complete
Selected K: 9
Test metrics: {'top1_acc': 0.9561847239865816, 'macro_recall': 0.9561454248064927, 'weighted_f1': 0.9562344505051618}
Index artifacts saved in: cesnet-quic22-w45-w46-w47-washed-knn\knn_index

% curl -X POST 127.0.0.1:8080/knn/index/load -H "Content-Type: application/json" -d '{"index_path": "cesnet-quic22-w45-w46-w47-washed-knn/knn_index/index.faiss", "labels_path": "cesnet-quic22-w45-w46-w47-washed-knn/knn_index/labels.npy"}'
{"message":"Index loaded","index_size":1051681,"index_dim":448,"metric":"cosine"}

> D:/Files/scoop/persist/miniforge/cuda128/python.exe g:/tc-transfer/test.py
Loaded 5000 valid samples from G:\cesnet-quic22\w44_washed_sample.csv (dropped 0).
Prepared 5000 samples into 40 requests (batch_size=128, workers=16).
Health: {'status': 'ok', 'details': {'device': 'cpu', 'feature_mode': 'backbone_gem', 'embedding_dim': 448, 'knn_ready': True, 'knn_size': 1051681, 'knn_metric': 'cosine'}}
Running warm-up requests: 5
Starting concurrent benchmark...
Progress: 4/40 requests completed
Progress: 8/40 requests completed
Progress: 12/40 requests completed
Progress: 16/40 requests completed
Progress: 20/40 requests completed
Progress: 24/40 requests completed
Progress: 28/40 requests completed
Progress: 32/40 requests completed
Progress: 36/40 requests completed
Progress: 40/40 requests completed

===== Load Test Report =====
Requests: total=40 success=40 failed=0 success_rate=100.00% throughput=0.45 req/s
Samples: total=5000 successful=5000 throughput=56.85 samples/s
Accuracy: top1=88.32% topK_hit=94.12%
Request latency(ms): mean=29176.01 p50=35499.75 p90=35739.97 p95=35813.32 p99=36072.86
Endpoint elapsed(ms): mean=29146.93 p50=35488.36 p90=35730.36 p95=35800.17 p99=36017.30
"""