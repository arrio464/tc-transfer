# High-Performance Inference Backend

This is a clean, standalone inference backend implementation for traffic classification embeddings and kNN prediction.

## Design goals

- Keep backend code isolated from the original experiment structure.
- Optimize for high-throughput embedding extraction and low-latency kNN search.
- Provide production-friendly API endpoints and artifact loading.
- Provide post-training scripts that produce backend-ready artifacts.

## API endpoints

- `GET /health`
  - Runtime status: model, device, embedding dimension, and kNN index state.
- `POST /embed`
  - Input: packet flows `[batch, 3, 30]`
  - Output: embeddings.
- `POST /knn/index/load`
  - Load a FAISS index and labels file at runtime.
- `POST /knn/search_by_embedding`
  - Query kNN directly with embeddings.
- `POST /knn/search`
  - Deprecated compatibility alias of `/knn/search_by_embedding`.
- `POST /predict/knn`
  - End-to-end: flow -> embedding -> kNN prediction.

## Run backend

1. Install dependencies:

   `pip install -r requirements/inference-backend.txt`

2. Start service:

   `python -m inference_backend.main --host 0.0.0.0 --port 8080`

3. Optional environment variables:

- `TC_BACKEND_DEVICE` (example: `cuda:0`, `cpu`)
- `TC_BACKEND_FEATURE_MODE` (`original` or `backbone_gem`)
- `TC_BACKEND_BATCH_SIZE`
- `TC_BACKEND_INDEX_METRIC` (`cosine` or `l2`)
- `TC_BACKEND_INDEX_PATH` + `TC_BACKEND_LABELS_PATH` for startup index loading

## Post-training scripts

### Full-parameter finetuning

`python -m post_training.finetune_full --dataset-npz /path/to/dataset.npz --output-dir /path/to/artifacts --feature-mode backbone_gem --epochs 50 --batch-size 512 --lr 3e-4 --amp`

Outputs:

- `best_finetune.pt`
- `label_classes.npy`
- `metrics.json`

### kNN transfer and index export

`python -m post_training.knn_transfer --dataset-npz /path/to/dataset.npz --output-dir /path/to/knn_artifacts --feature-mode backbone_gem --metric cosine --top-k 5 --auto-tune-k --faiss-gpu`

Outputs:

- `knn_index/index.faiss`
- `knn_index/labels.npy`
- `knn_index/meta.json`
- `metrics.json`

These index artifacts can be loaded using `POST /knn/index/load` or via startup env variables.

### Build NPZ from CESNET QUIC datasets

You can convert CESNET-QUIC22 (CSV) or CESNET-QUICEXT-25 (Parquet) into the NPZ structure expected by post-training scripts:

`python -m post_training.build_npz_from_cesnet --input-path /path/to/CESNET-QUIC22.csv --output-npz /path/to/quic22.npz --label-column QUIC_SNI --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --min-class-count 3 --max-packets 30`

`python -m post_training.build_npz_from_cesnet --input-path /path/to/CESNET-QUICEXT-25.parquet --output-npz /path/to/quicext25.npz --label-column QUIC_SNI --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --min-class-count 3 --max-packets 30`

Notes:

- The script reads only two columns: label column and `PPI`.
- It parses `PPI` in the canonical order: `[ipt, direction, packet_size]`.
- It truncates or zero-pads packet sequences to `[3, 30]` by default.
- It drops rows with missing label/PPI and labels with low frequency (`--min-class-count`).
- It writes `*.summary.json` with class/filter statistics.

### Build NPZ from CESNET QUIC datasets

You can convert CESNET-QUIC22 (CSV) or CESNET-QUICEXT-25 (Parquet) into the NPZ structure expected by post-training scripts:

`python -m post_training.build_npz_from_cesnet --input-path /path/to/CESNET-QUIC22.csv --output-npz /path/to/quic22.npz --label-column QUIC_SNI --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --min-class-count 3 --max-packets 30`

`python -m post_training.build_npz_from_cesnet --input-path /path/to/CESNET-QUICEXT-25.parquet --output-npz /path/to/quicext25.npz --label-column QUIC_SNI --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --min-class-count 3 --max-packets 30`

Notes:

- The script reads only two columns: label column and `PPI`.
- It parses `PPI` in the canonical order: `[ipt, direction, packet_size]`.
- It truncates or zero-pads packet sequences to `[3, 30]` by default.
- It drops rows with missing label/PPI and labels with low frequency (`--min-class-count`).
- It writes `*.summary.json` with class/filter statistics.

## NPZ dataset format expected by scripts

Required keys:

- Train flows: one of `train_flows`, `train_data`, `x_train`, `train_x`
- Train labels: one of `train_labels`, `y_train`, `train_y`
- Test flows: one of `test_flows`, `test_data`, `x_test`, `test_x`
- Test labels: one of `test_labels`, `y_test`, `test_y`

Optional keys:

- Validation flows: one of `val_flows`, `val_data`, `x_val`, `val_x`
- Validation labels: one of `val_labels`, `y_val`, `val_y`

Flow shape must be `[N, 3, 30]` where channels are `[ipt, direction, packet_size]`.
