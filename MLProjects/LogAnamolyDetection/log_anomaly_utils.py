"""
Utilities for the log anomaly detection pipeline: block ID extraction,
metrics I/O, and optional plotting for triage and dashboards.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


BLOCK_ID_PATTERN = re.compile(r"(blk_-?\d+)")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_block_id(text: Any) -> str | None:
    if pd.isna(text):
        return None
    match = BLOCK_ID_PATTERN.search(str(text))
    return match.group(1) if match else None


def save_metrics_json(metrics: dict[str, Any], output_path: Path) -> None:
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def set_plot_style() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10


def plot_pca_projection(
    pca_df: pd.DataFrame,
    output_path: Path | None = None,
    title: str = "PCA projection of anomaly scores",
) -> None:
    import matplotlib.pyplot as plt

    set_plot_style()
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["anomaly_score"],
        cmap="coolwarm",
        alpha=0.75,
        s=35,
    )
    ax.scatter(
        pca_df.loc[pca_df["is_anomaly"] == 1, "PC1"],
        pca_df.loc[pca_df["is_anomaly"] == 1, "PC2"],
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        s=70,
        label="Predicted anomaly",
    )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    plt.colorbar(scatter, ax=ax, label="Anomaly score")
    plt.tight_layout()
    if output_path is not None:
        ensure_directory(output_path.parent)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_template_anomalies(
    scored_df: pd.DataFrame,
    output_path: Path | None = None,
    top_n: int = 12,
) -> None:
    import matplotlib.pyplot as plt

    if "EventTemplate" not in scored_df.columns:
        return

    set_plot_style()
    template_counts = (
        scored_df.loc[scored_df["is_anomaly"] == 1, "EventTemplate"]
        .value_counts()
        .head(top_n)
        .sort_values()
    )
    if template_counts.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    template_counts.plot(kind="barh", ax=ax, color="tomato")
    ax.set_title("Most frequent templates among predicted anomalies")
    ax.set_xlabel("Count")
    ax.set_ylabel("EventTemplate")
    plt.tight_layout()
    if output_path is not None:
        ensure_directory(output_path.parent)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_score_distribution(
    scored_df: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_plot_style()
    fig, ax = plt.subplots()
    sns.histplot(scored_df["anomaly_score"], bins=30, kde=True, ax=ax, color="steelblue")
    ax.axvline(
        scored_df["anomaly_threshold"].iloc[0],
        color="crimson",
        linestyle="--",
        linewidth=2,
        label="Threshold",
    )
    ax.set_title("Anomaly score distribution")
    ax.set_xlabel("Anomaly score")
    ax.legend(loc="best")
    plt.tight_layout()
    if output_path is not None:
        ensure_directory(output_path.parent)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_sequence_anomalies(
    scored_df: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(scored_df["SequenceID"], scored_df["anomaly_score"], color="slateblue", alpha=0.7)
    anomaly_df = scored_df.loc[scored_df["is_anomaly"] == 1]
    ax.scatter(
        anomaly_df["SequenceID"],
        anomaly_df["anomaly_score"],
        color="crimson",
        s=35,
        label="Predicted anomaly",
    )
    ax.set_title("Anomaly score by sequence")
    ax.set_xlabel("SequenceID")
    ax.set_ylabel("Anomaly score")
    ax.legend(loc="best")
    plt.tight_layout()
    if output_path is not None:
        ensure_directory(output_path.parent)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
