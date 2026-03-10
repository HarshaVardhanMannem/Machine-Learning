# Real-Time Log Anomaly Detection System

A **production-grade log anomaly detection system** for structured log streams. It trains on historical logs, persists a detector and threshold, and scores new log batches in near real time—suitable for continuous monitoring, micro-batch pipelines, or integration with log shippers and alerting.

---

## Overview

- **Batch training**: Ingest logs → feature engineering → train Isolation Forest → persist model, preprocessor, vectorizer, and threshold.
- **Real-time inference**: Load persisted artifacts once; score incoming log batches with `score_batch()` for low-latency anomaly flags and scores.
- **Hybrid evaluation**: When labels exist, reports precision, recall, F1, ROC-AUC; otherwise runs in unsupervised mode with clear metrics and rankings.
- **Operational outputs**: Top-N anomalies, score distribution, component/template breakdowns, and optional plots for triage and dashboards.

This repo uses the **HDFS structured log sample** (LogHub) for demonstration; the same pipeline applies to any structured log source with compatible columns.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Log ingestion  │────▶│  Feature engine   │────▶│  Anomaly model   │
│  (file / stream)│     │  (numeric + TF-IDF)│     │  (Isolation Forest)│
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│  Alerts /       │◀────│  score_batch()   │◀─────────────┘
│  dashboards     │     │  (real-time API) │
└─────────────────┘     └──────────────────┘
```

- **Training** (scheduled or on-demand): `run_pipeline()` loads logs, engineers features, fits the model, and writes artifacts to `artifacts/`.
- **Inference** (real-time / micro-batch): `load_artifacts()` loads the saved model and threshold; `score_batch(logs_df)` returns anomaly scores and binary flags for each row.

---

## Project structure

| Path | Purpose |
|------|--------|
| `log_anomaly_config.py` | Configuration, paths, and env overrides (`LOG_ANOMALY_DATA_DIR`, `LOG_ANOMALY_ARTIFACTS_DIR`) |
| `log_anomaly_pipeline.py` | Training (`run_pipeline`), inference (`load_artifacts`, `score_batch`), and data loading |
| `log_anomaly_utils.py` | Block ID parsing, metrics I/O, and optional plotting |
| `Log Anamoly Detetction.ipynb` | Tutorial notebook: EDA, features, training, evaluation, and production notes |
| `data/` | Input logs (e.g. HDFS CSV); optional labels |
| `artifacts/` | Persisted model, preprocessor, vectorizer, metrics (with threshold), and scored outputs |

---

## Data

- **Structured logs**: CSV with at least `LineId`, `Component`, `Content`, `EventId`, `EventTemplate`. Optional: `Date`, `Time`, `Pid`, `Level`.
- **Labels** (optional): CSV with `BlockId` and `Label` for hybrid evaluation.
- By default, the pipeline can **auto-download** the HDFS sample; override paths and URLs via config or env.

---

## Quick start

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn requests joblib
# Optional for notebook plots:
pip install matplotlib seaborn
```

### 2. Train and save artifacts (batch)

```bash
cd LogAnamolyDetection
python -c "
from log_anomaly_pipeline import run_pipeline
result = run_pipeline()
print('Anomalies:', result['metrics'].get('predicted_anomalies'))
print('Threshold:', result['metrics'].get('anomaly_threshold'))
"
```

This downloads (if needed) and loads logs, trains the detector, and writes to `artifacts/`.

### 3. Score new logs (real-time / micro-batch)

```python
import pandas as pd
from log_anomaly_pipeline import load_artifacts, score_batch

# Load once (e.g. at service startup)
artifacts = load_artifacts()

# Score each incoming batch
new_logs = pd.read_csv("path/to/new_logs.csv")  # same schema as training
scored = score_batch(new_logs, artifacts=artifacts)
anomalies = scored[scored["is_anomaly"] == 1]
print(anomalies[["LineId", "Component", "EventTemplate", "anomaly_score"]])
```

Use `score_batch(logs_df, artifacts_dir=Path("artifacts"))` if you prefer loading from disk per call instead of reusing an in-memory `artifacts` dict.

### 4. Run on large volume

To stress-test or run on a large dataset, use `run_large_volume.py`. It builds a big CSV by replicating the sample (or your source file) to the target row count, then runs the full pipeline.

```bash
cd LogAnamolyDetection
# 100k rows (default)
python run_large_volume.py --target-rows 100000

# 500k rows, cap in-memory training at 100k to save RAM
python run_large_volume.py --target-rows 500000 --max-training-rows 100000
```

For very large CSVs you already have, set `max_training_rows` in config (or `get_default_config(max_training_rows=100000)`) so the pipeline samples down and avoids OOM.

---

## Configuration

- **Environment**: Set `LOG_ANOMALY_DATA_DIR` and `LOG_ANOMALY_ARTIFACTS_DIR` to override data and artifact paths (e.g. in Docker or cron).
- **Code**: `get_default_config(contamination=0.1, artifacts_dir=Path("/app/artifacts"), ...)` for programmatic overrides.
- Key options: `contamination`, `max_training_rows` (cap rows for huge CSVs), `top_n_anomalies`, `max_tfidf_features`, `use_dbscan_comparison`; see `log_anomaly_config.py`.

---

## Modeling

- **Primary detector**: Isolation Forest on combined numeric + one-hot + TF-IDF features.
- **Threshold**: Chosen from training scores (e.g. top `contamination` fraction); stored in `artifacts/metrics.json` and used in `score_batch`.
- **Optional**: DBSCAN comparison for agreement rate and analysis; not required for real-time scoring.
- **Features**: Sequence and template diversity, component/level frequencies, block IDs, content length, TF-IDF on log content.

---

## Evaluation and interpretation

- **With labels**: Precision, recall, F1, ROC-AUC, PR-AUC, and classification report.
- **Without labels**: Unsupervised metrics, anomaly rate, detector agreement, and ranked top-N list.
- **Operational**: Top anomalies CSV, score distribution, and (in the notebook) PCA and template/component anomaly plots.

---

## Production and deployment

- **Logging**: The pipeline uses Python `logging`; configure level and handlers in your app (e.g. `logging.basicConfig(level=logging.INFO)`).
- **Deployment**: Run training as a scheduled job; run inference in a small service or script that calls `score_batch()` on each micro-batch.
- **Monitoring**: Track score distribution, anomaly rate, and (if applicable) schema/template drift; retrain when data distribution changes.

---

## Limitations

- **Schema**: Input logs must match the expected columns; validate upstream.
- **Cold start**: First run needs enough history to train; `score_batch` is for scoring after at least one successful `run_pipeline`.
- **Threshold**: Fixed from training; for evolving streams, consider periodic retraining or dynamic threshold logic.
- **Scale**: Designed for single-machine, in-memory batches; for very high throughput, consider batching and/or distributed inference.

---

## Run the notebook

Open `Log Anamoly Detetction.ipynb` and run all cells. It walks through data load, EDA, feature engineering, training, evaluation, and artifact usage with production notes.
