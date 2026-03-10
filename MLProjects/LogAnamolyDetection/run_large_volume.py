"""
Run the anomaly detection pipeline on a large volume of input by replicating
the HDFS sample (or your data) to the desired size. Use for stress-testing
and high-volume runs.

Example (100k rows):
    python run_large_volume.py --target-rows 100000

Example (500k rows, cap training at 100k to save memory):
    python run_large_volume.py --target-rows 500000 --max-training-rows 100000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_anomaly_config import DATA_DIR, get_default_config
from log_anomaly_pipeline import load_logs, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_large_csv(
    source_path: Path,
    output_path: Path,
    target_rows: int,
    random_state: int = 42,
) -> Path:
    """Replicate source CSV to reach target_rows; LineIds are made unique."""
    df = pd.read_csv(source_path)
    n = len(df)
    if n >= target_rows:
        df = df.sample(n=target_rows, random_state=random_state).reset_index(drop=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Wrote %d rows to %s", len(df), output_path)
        return output_path

    replications = (target_rows + n - 1) // n
    chunks = []
    for i in range(replications):
        part = df.copy()
        part["LineId"] = part["LineId"].astype(int) + (i * (10 ** (len(str(target_rows)) + 1)))
        chunks.append(part)
    out = pd.concat(chunks, ignore_index=True)
    if len(out) > target_rows:
        out = out.sample(n=target_rows, random_state=random_state).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info("Built %d rows from %d replications -> %s", len(out), replications, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anomaly detection on large log volume")
    parser.add_argument(
        "--target-rows",
        type=int,
        default=100_000,
        help="Target number of log rows (default: 100000)",
    )
    parser.add_argument(
        "--max-training-rows",
        type=int,
        default=None,
        help="Cap training set size (sample if source is larger); default: use all",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA_DIR / "HDFS_2k.log_structured.csv",
        help="Source structured CSV to replicate (default: data/HDFS_2k.log_structured.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "HDFS_large.log_structured.csv",
        help="Output path for large CSV (default: data/HDFS_large.log_structured.csv)",
    )
    args = parser.parse_args()

    if not args.source.exists():
        logger.info("Source not found; loading via pipeline to trigger download")
        cfg = get_default_config()
        load_logs(cfg)
        args.source = cfg.structured_logs_path
    if not args.source.exists():
        logger.error("Source CSV not found at %s", args.source)
        sys.exit(1)

    build_large_csv(
        args.source,
        args.output,
        target_rows=args.target_rows,
        random_state=get_default_config().random_state,
    )
    config = get_default_config(
        structured_logs_path=args.output,
        max_training_rows=args.max_training_rows,
    )
    logger.info("Running pipeline on %s (max_training_rows=%s)", args.output, config.max_training_rows)
    result = run_pipeline(config)
    m = result["metrics"]
    logger.info(
        "Done. Rows=%s, Anomalies=%s, Threshold=%s",
        m.get("row_count"),
        m.get("predicted_anomalies"),
        m.get("anomaly_threshold"),
    )


if __name__ == "__main__":
    main()
