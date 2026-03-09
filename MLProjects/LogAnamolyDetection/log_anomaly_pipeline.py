from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from log_anomaly_config import PipelineConfig, get_default_config
from log_anomaly_utils import ensure_directory, extract_block_id, save_metrics_json


ARTIFACT_MODEL_NAME = "isolation_forest.joblib"
ARTIFACT_PREPROCESSOR_NAME = "preprocessor.joblib"


def download_file(url: str, destination: Path) -> Path:
    ensure_directory(destination.parent)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def load_logs(config: PipelineConfig | None = None) -> pd.DataFrame:
    config = config or get_default_config()
    if not config.structured_logs_path.exists():
        if not config.auto_download_structured_logs:
            raise FileNotFoundError(f"Missing structured logs at {config.structured_logs_path}")
        download_file(config.structured_logs_url, config.structured_logs_path)

    logs_df = pd.read_csv(config.structured_logs_path)
    validate_schema(logs_df, config)
    logs_df = logs_df.copy()
    logs_df["SequenceID"] = np.arange(len(logs_df))
    logs_df["BlockId"] = logs_df["Content"].map(extract_block_id)
    logs_df["HasBlockId"] = logs_df["BlockId"].notna().astype(int)
    logs_df["Content"] = logs_df["Content"].fillna("").astype(str)
    logs_df["ContentLength"] = logs_df["Content"].str.len()
    logs_df["TokenCount"] = logs_df["Content"].str.split().str.len()
    return logs_df


def load_optional_labels(config: PipelineConfig | None = None) -> pd.DataFrame | None:
    config = config or get_default_config()
    if not config.labels_path.exists():
        if not config.auto_download_labels:
            return None
        try:
            download_file(config.labels_url, config.labels_path)
        except requests.RequestException:
            return None

    labels_df = pd.read_csv(config.labels_path)
    expected_columns = {"BlockId", "Label"}
    if not expected_columns.issubset(labels_df.columns):
        return None
    return labels_df


def validate_schema(logs_df: pd.DataFrame, config: PipelineConfig) -> None:
    missing_required = [col for col in config.required_columns if col not in logs_df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")


def attach_labels(logs_df: pd.DataFrame, labels_df: pd.DataFrame | None) -> pd.DataFrame:
    logs_df = logs_df.copy()
    if labels_df is None:
        logs_df["Label"] = "Unknown"
        logs_df["HasGroundTruth"] = 0
        return logs_df

    merged_df = logs_df.merge(labels_df[["BlockId", "Label"]], on="BlockId", how="left")
    merged_df["Label"] = merged_df["Label"].fillna("Unknown")
    merged_df["HasGroundTruth"] = (merged_df["Label"] != "Unknown").astype(int)
    return merged_df


def engineer_features(logs_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = logs_df.copy()
    feature_df["EventTemplateCode"] = feature_df["EventTemplate"].astype("category").cat.codes
    feature_df["TemplateFrequency"] = feature_df.groupby("EventTemplate")["LineId"].transform("count")
    feature_df["ComponentFrequency"] = feature_df.groupby("Component")["LineId"].transform("count")
    feature_df["LevelFrequency"] = feature_df.groupby("Level")["LineId"].transform("count")
    feature_df["BlockFrequency"] = feature_df.groupby("BlockId")["LineId"].transform("count").fillna(0)
    feature_df["IsRareTemplate"] = (
        feature_df["TemplateFrequency"] <= feature_df["TemplateFrequency"].quantile(0.10)
    ).astype(int)

    feature_df["PrevSameTemplate"] = (
        feature_df["EventTemplate"] == feature_df["EventTemplate"].shift(1)
    ).fillna(False).astype(int)
    feature_df["PrevSameComponent"] = (
        feature_df["Component"] == feature_df["Component"].shift(1)
    ).fillna(False).astype(int)

    rolling_template_uniques = (
        feature_df["EventTemplateCode"]
        .rolling(window=20, min_periods=1)
        .apply(lambda values: pd.Series(values).nunique(), raw=False)
    )
    feature_df["WindowTemplateUniques"] = rolling_template_uniques.fillna(0)
    feature_df["SequenceDelta"] = feature_df["SequenceID"].diff().fillna(0)

    return feature_df


def get_model_columns(feature_df: pd.DataFrame) -> dict[str, list[str]]:
    categorical_cols = ["EventTemplate", "EventId", "Component", "Level"]
    if "BlockId" in feature_df.columns:
        categorical_cols.append("BlockId")

    numeric_cols = [
        "SequenceID",
        "EventTemplateCode",
        "Pid",
        "ContentLength",
        "TokenCount",
        "TemplateFrequency",
        "ComponentFrequency",
        "LevelFrequency",
        "BlockFrequency",
        "HasBlockId",
        "IsRareTemplate",
        "PrevSameTemplate",
        "PrevSameComponent",
        "WindowTemplateUniques",
        "SequenceDelta",
    ]
    categorical_cols = [col for col in categorical_cols if col in feature_df.columns]
    numeric_cols = [col for col in numeric_cols if col in feature_df.columns]
    return {"categorical": categorical_cols, "numeric": numeric_cols}


def build_preprocessor(feature_df: pd.DataFrame, config: PipelineConfig) -> tuple[ColumnTransformer, list[str], list[str]]:
    model_cols = get_model_columns(feature_df)
    categorical_cols = model_cols["categorical"]
    numeric_cols = model_cols["numeric"]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_cols),
            ("numeric", numeric_transformer, numeric_cols),
        ],
        remainder="drop",
    )
    return preprocessor, categorical_cols, numeric_cols


def combine_model_features(
    feature_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    config: PipelineConfig,
) -> tuple[sparse.csr_matrix, TfidfVectorizer]:
    structured_matrix = preprocessor.fit_transform(feature_df)
    vectorizer = TfidfVectorizer(
        max_features=config.max_tfidf_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    text_matrix = vectorizer.fit_transform(feature_df["Content"])
    combined_matrix = sparse.hstack([structured_matrix, text_matrix]).tocsr()
    return combined_matrix, vectorizer


def score_anomalies(
    feature_df: pd.DataFrame,
    combined_matrix: sparse.csr_matrix,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, IsolationForest, np.ndarray]:
    model = IsolationForest(
        contamination=config.contamination,
        random_state=config.random_state,
        n_estimators=250,
    )
    model.fit(combined_matrix)

    raw_scores = -model.score_samples(combined_matrix)
    threshold = float(np.quantile(raw_scores, 1 - config.contamination))
    is_anomaly = (raw_scores >= threshold).astype(int)

    scored_df = feature_df.copy()
    scored_df["anomaly_score"] = raw_scores
    scored_df["anomaly_threshold"] = threshold
    scored_df["is_anomaly"] = is_anomaly
    scored_df["rank"] = scored_df["anomaly_score"].rank(method="dense", ascending=False).astype(int)
    return scored_df, model, raw_scores


def add_dbscan_comparison(
    scored_df: pd.DataFrame,
    combined_matrix: sparse.csr_matrix,
    config: PipelineConfig,
) -> pd.DataFrame:
    scored_df = scored_df.copy()
    if not config.use_dbscan_comparison:
        scored_df["dbscan_is_anomaly"] = 0
        scored_df["detector_agreement"] = scored_df["is_anomaly"]
        return scored_df

    dense_matrix = combined_matrix.toarray()
    dbscan = DBSCAN(eps=config.dbscan_eps, min_samples=config.dbscan_min_samples)
    dbscan_labels = dbscan.fit_predict(dense_matrix)
    dbscan_is_anomaly = (dbscan_labels == -1).astype(int)
    scored_df["dbscan_is_anomaly"] = dbscan_is_anomaly
    scored_df["detector_agreement"] = (
        scored_df["is_anomaly"] == scored_df["dbscan_is_anomaly"]
    ).astype(int)
    return scored_df


def evaluate_predictions(scored_df: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "row_count": int(len(scored_df)),
        "predicted_anomalies": int(scored_df["is_anomaly"].sum()),
        "predicted_anomaly_rate": float(scored_df["is_anomaly"].mean()),
        "detector_agreement_rate": float(scored_df.get("detector_agreement", pd.Series([1])).mean()),
    }

    label_candidates = scored_df.loc[scored_df["HasGroundTruth"] == 1].copy()
    if label_candidates.empty:
        metrics["evaluation_mode"] = "unsupervised_only"
        return metrics

    label_candidates["ground_truth"] = (
        label_candidates["Label"].astype(str).str.lower().isin({"anomaly", "anomalous", "1"})
    ).astype(int)
    if label_candidates["ground_truth"].nunique() < 2:
        metrics["evaluation_mode"] = "labels_present_but_not_binary"
        return metrics

    y_true = label_candidates["ground_truth"].to_numpy()
    y_pred = label_candidates["is_anomaly"].to_numpy()
    y_score = label_candidates["anomaly_score"].to_numpy()

    metrics.update(
        {
            "evaluation_mode": "hybrid_with_ground_truth",
            "ground_truth_rows": int(len(label_candidates)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "average_precision": float(average_precision_score(y_true, y_score)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "classification_report": classification_report(
                y_true,
                y_pred,
                zero_division=0,
                output_dict=True,
            ),
        }
    )
    return metrics


def build_pca_frame(
    scored_df: pd.DataFrame,
    combined_matrix: sparse.csr_matrix,
    config: PipelineConfig,
) -> pd.DataFrame:
    pca = PCA(n_components=config.pca_components, random_state=config.random_state)
    reduced = pca.fit_transform(combined_matrix.toarray())
    pca_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    pca_df["anomaly_score"] = scored_df["anomaly_score"].values
    pca_df["is_anomaly"] = scored_df["is_anomaly"].values
    pca_df["SequenceID"] = scored_df["SequenceID"].values
    return pca_df


def save_artifacts(
    scored_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    model: IsolationForest,
    metrics: dict[str, Any],
    config: PipelineConfig,
) -> None:
    ensure_directory(config.artifacts_dir)
    scored_df.to_csv(config.scored_logs_path, index=False)
    scored_df.nlargest(config.top_n_anomalies, "anomaly_score").to_csv(
        config.top_anomalies_path,
        index=False,
    )
    save_metrics_json(metrics, config.metrics_path)
    joblib.dump(model, config.artifacts_dir / ARTIFACT_MODEL_NAME)
    joblib.dump(preprocessor, config.artifacts_dir / ARTIFACT_PREPROCESSOR_NAME)


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, Any]:
    config = config or get_default_config()
    logs_df = load_logs(config)
    labels_df = load_optional_labels(config)
    labeled_logs_df = attach_labels(logs_df, labels_df)
    feature_df = engineer_features(labeled_logs_df)
    preprocessor, _, _ = build_preprocessor(feature_df, config)
    combined_matrix, vectorizer = combine_model_features(feature_df, preprocessor, config)
    scored_df, model, _ = score_anomalies(feature_df, combined_matrix, config)
    scored_df = add_dbscan_comparison(scored_df, combined_matrix, config)
    metrics = evaluate_predictions(scored_df)
    pca_df = build_pca_frame(scored_df, combined_matrix, config)
    save_artifacts(scored_df, preprocessor, model, metrics, config)

    return {
        "config": asdict(config),
        "logs_df": logs_df,
        "feature_df": feature_df,
        "scored_df": scored_df,
        "metrics": metrics,
        "pca_df": pca_df,
        "preprocessor": preprocessor,
        "vectorizer": vectorizer,
        "model": model,
        "labels_df": labels_df,
    }
