## Log Anomaly Detection (HDFS) – Production-Style Demo

This project implements a production-style **log anomaly detection** workflow on the structured HDFS sample from the LogHub dataset.

It includes:
- a reusable Python anomaly pipeline
- a documented Jupyter notebook
- saved artifacts (scored logs, top anomalies, metrics, serialized model/preprocessor)

---

## Project structure

- `Log Anamoly Detetction.ipynb` – main tutorial notebook
- `log_anomaly_config.py` – configuration and paths for data, artifacts, and model hyperparameters
- `log_anomaly_pipeline.py` – end-to-end anomaly detection pipeline (data load → features → model → scores → artifacts)
- `log_anomaly_utils.py` – utilities for block ID extraction, metrics saving, and plotting
- `data/` – structured HDFS log CSV and optional label files
- `artifacts/` – generated outputs (scored logs, top anomalies, metrics JSON, model artifacts)

---

## Data

The pipeline uses the **HDFS structured logs** sample from LogHub:
- structured CSV: `HDFS_2k.log_structured.csv`
- optional labels: `anomaly_label.csv` (block-level anomaly labels)

By default:
- if `data/HDFS_2k.log_structured.csv` is missing, it is **downloaded automatically**
- labels are **optional** – the pipeline runs fully in unsupervised mode if they are not present or download is disabled

Key columns:
- `LineId`, `Date`, `Time`, `Pid`, `Level`, `Component`
- `Content` (raw log message)
- `EventId`, `EventTemplate`

The pipeline also:
- extracts `BlockId` from `Content` when present
- adds `SequenceID` to preserve log order

---

## Environment & dependencies

Required Python packages (typical environment):

- `pandas`
- `numpy`
- `scikit-learn`
- `requests`
- `matplotlib` (for plotting in the notebook)
- `seaborn` (for plotting in the notebook)
- `joblib`

Install (example with `pip`):

```bash
pip install pandas numpy scikit-learn requests matplotlib seaborn joblib
```

The core pipeline code (`run_pipeline`) does **not** require Jupyter; plotting in the notebook uses `matplotlib`/`seaborn` if available.

---

## How to run

### 1. Run as a Python pipeline

From the `LogAnamolyDetection` folder:

```bash
cd "c:\Users\harsh\MLProjects\ML concepts\MLProjects\LogAnamolyDetection"

python -c "from log_anomaly_pipeline import run_pipeline; result = run_pipeline(); print(result['metrics'])"
```

This will:
- download/load HDFS structured logs (if not present)
- attach labels if available
- engineer structured + TF‑IDF features
- train an **Isolation Forest** model
- optionally compute a **DBSCAN** comparison signal
- score and rank anomalies
- compute hybrid evaluation metrics (unsupervised-only if no labels)
- save artifacts under `artifacts/`

Generated artifacts include:
- `artifacts/scored_logs.csv` – all logs with anomaly scores and flags
- `artifacts/top_anomalies.csv` – highest-scoring anomalies for triage
- `artifacts/metrics.json` – summary metrics (mode, counts, evaluation stats where labels exist)
- `artifacts/isolation_forest.joblib` – trained Isolation Forest model
- `artifacts/preprocessor.joblib` – fitted preprocessing pipeline

### 2. Run the notebook

Open:

- `Log Anamoly Detetction.ipynb`

and run all cells from top to bottom.

The notebook walks through:
- problem framing and dataset description
- schema checks and basic EDA
- feature engineering choices and rationale
- model training and anomaly scoring
- ranked anomalies and hybrid evaluation
- PCA and auxiliary plots (if plotting libraries are installed)
- production notes and limitations

---

## Modeling approach

Core detector:
- **Isolation Forest** is used as the primary anomaly detector.

Comparison:
- **DBSCAN** can be enabled as a secondary view of outliers; its binary flags and agreement rate with Isolation Forest are exposed in the results for analysis, but it is not the primary production signal.

Feature engineering:
- numeric features: `SequenceID`, content length, token count, template/component/level frequencies, block frequency, rare-template flag, previous-template/component continuity, rolling template diversity, etc.
- categorical features: event template, event ID, component, level, block ID (when available), one-hot encoded
- text features: TF‑IDF over log content

The combined feature matrix (structured + TF‑IDF) feeds Isolation Forest.

---

## Evaluation and interpretation

Hybrid evaluation:
- if labels are available, metrics include precision, recall, F1, ROC‑AUC, PR‑AUC, and a detailed classification report
- if labels are not available, the metrics clearly report **unsupervised-only** mode and focus on counts, rates, and detector agreement

Interpretability:
- top‑ranked anomalies with scores and key fields (component, level, template, block ID)
- anomaly concentration by template and component
- anomaly score distribution and sequence‑by‑sequence plots
- PCA projection of anomalies when plotting is enabled

---

## Production notes & limitations

This project is designed as a **practical starting point**, not a complete production system:

- schema validation and data-quality checks should be integrated with ingestion
- thresholds and contamination rates should be tuned per environment and use case
- monitoring should include score drift, template drift, and data volume changes
- session/block/window-level aggregation is often necessary to capture multi-line failures
- labels in real systems are often delayed or partial; treat metrics as guidance, not absolute truth

The modular structure (config, pipeline, utils) is intended to make it easy to:
- deploy the pipeline as a scheduled batch job
- export scores into an alerting or dashboarding system
- iterate on features and model configuration without rewriting the notebook

