# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `CONTRIBUTING.md` — contributor guide covering setup, notebook standards, project standards, workflow, and PR checklist
- `CHANGELOG.md` — this file
- `LICENSE` — MIT license file
- Production-grade root `README.md` with badges, Table of Contents, Contributing & Acknowledgements sections

---

## [1.2.0] — 2025-04-01

### Added
- **Euclidean vs. Manhattan Distance** concept notebook (`ML concepts/Euclidean_vs_Manhattan_Distance_ML.ipynb`)
  - Geometric intuition visualisations
  - KNN decision boundary comparison under both metrics
  - K-Means (Euclidean) vs. K-Medians (Manhattan) clustering comparison
  - Practical metric selection decision matrix

---

## [1.1.0] — 2025-03-01

### Added
- **Log Anomaly Detection** end-to-end project (`MLProjects/LogAnomalyDetection/`)
  - Isolation Forest trained on numeric + one-hot + TF-IDF features from HDFS logs
  - `score_batch()` API for real-time / micro-batch inference
  - Hybrid supervised/unsupervised evaluation (Precision, Recall, F1, ROC-AUC, PR-AUC)
  - Artifact persistence (`artifacts/`) for deployment
  - Optional DBSCAN comparison

---

## [1.0.0] — 2025-01-01

### Added
- **CPU vs. GPU Performance Benchmark** notebook (`ML concepts/CPU_vs_GPU.ipynb`)
- **Standardization in ML** notebook (`ML concepts/Standardization in ML.ipynb`)
- **Normalization in ML** notebook (`ML concepts/Normalization in ML.ipynb`)
- **Classification Metrics Demo** notebook (`ML concepts/Accuracy_Precision_Recall_F1 score  in ML Demo.ipynb`)
- Initial repository structure with `ML concepts/` and `MLProjects/` sections

[Unreleased]: https://github.com/HarshaVardhanMannem/Machine-Learning/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/HarshaVardhanMannem/Machine-Learning/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/HarshaVardhanMannem/Machine-Learning/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/HarshaVardhanMannem/Machine-Learning/releases/tag/v1.0.0
