# 🚀 ML Projects

> End-to-end machine learning projects that go beyond concept demonstrations. Each project includes production-oriented design, reusable pipelines, and real-world datasets.

---

## 📂 Projects

### 1. 🔍 Log Anomaly Detection
**Folder:** [`LogAnamolyDetection/`](LogAnamolyDetection/)

A **production-grade log anomaly detection system** for structured log streams. It trains on historical logs, persists a detector and threshold, and scores new log batches in near real time — suitable for continuous monitoring, micro-batch pipelines, or integration with log shippers and alerting tools.

**Highlights:**
- Trains an **Isolation Forest** on combined numeric + one-hot + TF-IDF features extracted from structured HDFS logs
- Supports **real-time / micro-batch inference** via a `score_batch()` API — load artifacts once, score each incoming batch with low latency
- **Hybrid evaluation**: when ground-truth labels are available, reports Precision, Recall, F1, ROC-AUC, and PR-AUC; otherwise runs in unsupervised mode
- Persists model, preprocessor, vectorizer, threshold, and scored outputs to `artifacts/` for reproducibility and deployment
- Optional **DBSCAN** comparison for anomaly-rate cross-validation

**Tech:** Python · scikit-learn · pandas · NumPy · joblib · Matplotlib/Seaborn (optional)

📖 [Full documentation →](LogAnamolyDetection/README.md)

---

## 🛠️ General Prerequisites

```bash
pip install pandas numpy scikit-learn requests joblib matplotlib seaborn jupyter
```

Each project folder contains its own `README.md` with specific setup, usage, and configuration instructions.

---

## 💡 Design Principles Across Projects

- **Production-oriented** — pipelines are designed to be scheduled, containerised, or deployed as microservices
- **Reproducibility** — artifacts and configurations are persisted so results can be recreated
- **Modularity** — feature engineering, training, and inference are separated into distinct components
- **Observability** — metrics, top-N anomaly lists, and score distributions are exported for dashboards and alerting
