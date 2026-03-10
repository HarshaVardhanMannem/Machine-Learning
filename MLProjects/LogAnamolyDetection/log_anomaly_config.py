"""
Configuration for the log anomaly detection pipeline and real-time inference.

Supports environment overrides for deployment:
- LOG_ANOMALY_DATA_DIR: data directory (default: project/data)
- LOG_ANOMALY_ARTIFACTS_DIR: artifact output directory (default: project/artifacts)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent
_DATA_DIR_ENV = os.environ.get("LOG_ANOMALY_DATA_DIR")
_ARTIFACTS_DIR_ENV = os.environ.get("LOG_ANOMALY_ARTIFACTS_DIR")
DATA_DIR = Path(_DATA_DIR_ENV) if _DATA_DIR_ENV else BASE_DIR / "data"
ARTIFACTS_DIR = Path(_ARTIFACTS_DIR_ENV) if _ARTIFACTS_DIR_ENV else BASE_DIR / "artifacts"

HDFS_STRUCTURED_URL = (
    "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/"
    "HDFS_2k.log_structured.csv"
)
HDFS_LABELS_URL = (
    "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/"
    "anomaly_label.csv"
)


@dataclass
class PipelineConfig:
    """Central configuration for training and inference; paths respect LOG_ANOMALY_* env vars."""

    random_state: int = 42
    contamination: float = 0.05
    max_training_rows: Optional[int] = None  # If set, sample this many rows when input is larger (for huge CSVs)
    max_tfidf_features: int = 80
    use_dbscan_comparison: bool = True
    dbscan_eps: float = 1.25
    dbscan_min_samples: int = 8
    top_n_anomalies: int = 25
    pca_components: int = 2

    data_dir: Path = DATA_DIR
    artifacts_dir: Path = ARTIFACTS_DIR
    structured_logs_path: Path = DATA_DIR / "HDFS_2k.log_structured.csv"
    labels_path: Path = DATA_DIR / "anomaly_label.csv"
    scored_logs_path: Path = ARTIFACTS_DIR / "scored_logs.csv"
    top_anomalies_path: Path = ARTIFACTS_DIR / "top_anomalies.csv"
    metrics_path: Path = ARTIFACTS_DIR / "metrics.json"
    pca_plot_path: Path = ARTIFACTS_DIR / "anomaly_pca.png"
    template_plot_path: Path = ARTIFACTS_DIR / "template_anomalies.png"
    score_plot_path: Path = ARTIFACTS_DIR / "score_distribution.png"
    sequence_plot_path: Path = ARTIFACTS_DIR / "sequence_anomalies.png"

    structured_logs_url: str = HDFS_STRUCTURED_URL
    labels_url: str = HDFS_LABELS_URL
    auto_download_structured_logs: bool = True
    auto_download_labels: bool = False

    required_columns: tuple[str, ...] = (
        "LineId",
        "Component",
        "Content",
        "EventId",
        "EventTemplate",
    )
    optional_columns: tuple[str, ...] = ("Date", "Time", "Pid", "Level")

    text_column_candidates: tuple[str, ...] = (
        "Content",
        "Message",
        "LogMessage",
        "EventTemplate",
    )

    metadata_columns: tuple[str, ...] = (
        "LineId",
        "Date",
        "Time",
        "Pid",
        "Level",
        "Component",
        "Content",
        "EventId",
        "EventTemplate",
        "BlockId",
        "Label",
    )

    feature_prefixes: tuple[str, ...] = (
        "tfidf_",
        "template_",
        "component_",
        "level_",
        "window_",
    )

    notes: dict[str, str] = field(
        default_factory=lambda: {
            "primary_model": "IsolationForest",
            "secondary_model": "DBSCAN",
            "dataset": "LogHub HDFS structured log sample",
        }
    )


def get_default_config(**overrides: Optional[object]) -> PipelineConfig:
    """Return default config; any overrides (e.g. artifacts_dir=Path(...)) are applied."""
    config = PipelineConfig()
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    return config
