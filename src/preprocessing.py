"""
Data Preprocessing Pipeline for Network Attack Forecasting Agent (SIH 2026 - SIH26153).

Ingests raw network flow logs (CICIDS2017, UNSW-NB15, or synthetic formats)
and aggregates them into regularized time-series windows with cybersecurity metrics:
- total_flows
- attack_count
- risk_score
- byte_volume
- distinct_src_ips
- syn_flag_count
- failed_connections
"""

import argparse
import logging
import os
import re
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from src.generate_mock_data import generate_mock_flows
except ImportError:
    from generate_mock_data import generate_mock_flows

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocessing")

# Benign/normal labels across benchmark datasets (case-insensitive)
BENIGN_LABELS = {"benign", "normal", "0", "0.0", "clean", "false"}


def normalize_column_name(col: str) -> str:
    """Strip whitespace, replace special characters, and lowercase column names."""
    cleaned = col.strip()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned.lower()


def map_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map heterogeneous column names from CICIDS2017, UNSW-NB15, or custom formats
    into a canonical schema:
    - timestamp
    - src_ip
    - dst_ip
    - fwd_bytes
    - bwd_bytes
    - syn_flag
    - failed_connection
    - label
    """
    col_map = {normalize_column_name(c): c for c in df.columns}
    normalized_df = pd.DataFrame(index=df.index)

    # 1. Timestamp mapping
    ts_candidates = [
        "timestamp", "time", "datetime", "stime", "start_time", "flow_start_time"
    ]
    matched_ts = None
    for cand in ts_candidates:
        if cand in col_map:
            matched_ts = col_map[cand]
            break

    if not matched_ts:
        raise ValueError(f"Could not find a valid timestamp column in: {list(df.columns)}")
    normalized_df["timestamp"] = df[matched_ts]

    # 2. Source IP mapping
    src_candidates = ["src_ip", "srcip", "source_ip", "sourceip", "src"]
    matched_src = None
    for cand in src_candidates:
        if cand in col_map:
            matched_src = col_map[cand]
            break
    normalized_df["src_ip"] = df[matched_src].astype(str) if matched_src else "192.168.1.1"

    # 3. Destination IP mapping
    dst_candidates = ["dst_ip", "dstip", "destination_ip", "destinationip", "dst"]
    matched_dst = None
    for cand in dst_candidates:
        if cand in col_map:
            matched_dst = col_map[cand]
            break
    normalized_df["dst_ip"] = df[matched_dst].astype(str) if matched_dst else "10.0.0.1"

    # 4. Forward Bytes mapping
    fwd_candidates = [
        "fwd_bytes", "total_length_of_fwd_packets", "sbytes", "fwd_pkt_len_tot", "bytes_in"
    ]
    matched_fwd = None
    for cand in fwd_candidates:
        if cand in col_map:
            matched_fwd = col_map[cand]
            break
    normalized_df["fwd_bytes"] = (
        pd.to_numeric(df[matched_fwd], errors="coerce").fillna(0).astype(np.int64)
        if matched_fwd else 0
    )

    # 5. Backward Bytes mapping
    bwd_candidates = [
        "bwd_bytes", "total_length_of_bwd_packets", "dbytes", "bwd_pkt_len_tot", "bytes_out"
    ]
    matched_bwd = None
    for cand in bwd_candidates:
        if cand in col_map:
            matched_bwd = col_map[cand]
            break
    normalized_df["bwd_bytes"] = (
        pd.to_numeric(df[matched_bwd], errors="coerce").fillna(0).astype(np.int64)
        if matched_bwd else 0
    )

    # Combined byte volume
    normalized_df["byte_volume"] = normalized_df["fwd_bytes"] + normalized_df["bwd_bytes"]

    # 6. SYN Flag Count mapping
    syn_candidates = ["syn_flag", "syn_flag_count", "syn_count", "synflag"]
    matched_syn = None
    for cand in syn_candidates:
        if cand in col_map:
            matched_syn = col_map[cand]
            break
    normalized_df["syn_flag"] = (
        pd.to_numeric(df[matched_syn], errors="coerce").fillna(0).astype(np.int64)
        if matched_syn else 0
    )

    # 7. Failed Connection mapping
    failed_candidates = [
        "failed_connection", "failed_connections", "failed_conn", "conn_failed", "is_failed"
    ]
    matched_failed = None
    for cand in failed_candidates:
        if cand in col_map:
            matched_failed = col_map[cand]
            break
    if matched_failed:
        normalized_df["failed_connection"] = (
            pd.to_numeric(df[matched_failed], errors="coerce").fillna(0).astype(np.int64)
        )
    else:
        # If no explicit failed flag, default to 0
        normalized_df["failed_connection"] = 0

    # 8. Label mapping & Attack identification
    label_candidates = ["label", "attack_cat", "attack", "class", "target"]
    matched_label = None
    for cand in label_candidates:
        if cand in col_map:
            matched_label = col_map[cand]
            break

    if matched_label:
        raw_labels = df[matched_label].astype(str).str.strip()
        normalized_df["raw_label"] = raw_labels
        # Determine is_attack: 1 if not benign/normal, else 0
        normalized_df["is_attack"] = raw_labels.apply(
            lambda x: 0 if str(x).lower() in BENIGN_LABELS else 1
        ).astype(np.int64)
    else:
        logger.warning("No label column detected; defaulting all flows to BENIGN (is_attack=0)")
        normalized_df["raw_label"] = "BENIGN"
        normalized_df["is_attack"] = 0

    return normalized_df


def parse_timestamps(series: pd.Series) -> pd.Series:
    """
    Robustly parse diverse timestamp formats into timezone-naive datetime objects.
    Supports Unix timestamps, standard ISO format, and day/month/year variants.
    """
    sample = series.dropna().iloc[0] if len(series.dropna()) > 0 else None
    if sample is None:
        return pd.to_datetime(series)

    # Check for Unix epoch (integers/floats e.g. 1499416200 or 1499416200.123)
    if isinstance(sample, (int, float, np.integer, np.floating)) or (
        isinstance(sample, str) and re.match(r"^\d{9,12}(\.\d+)?$", sample.strip())
    ):
        return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="s")

    # Try standard flexible datetime parser
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().sum() > len(parsed) * 0.5:
        # Fallback to dayfirst format (common in Australian UNSW / European datasets)
        parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")

    return parsed


class FlowDataPreprocessor:
    """
    End-to-end network flow preprocessor that loads raw traffic, normalizes schemas,
    and constructs regularized time-series aggregation windows.
    """

    def __init__(self, window_size: str = "5min"):
        """
        Args:
            window_size: Pandas frequency string for aggregation windows (e.g., '5min', '15min', '1h').
        """
        self.window_size = window_size

    def load_data(
        self,
        filepath: Optional[str] = None,
        auto_generate_mock: bool = True,
    ) -> pd.DataFrame:
        """
        Load network flow data from CSV. If missing and auto_generate_mock is True,
        generates synthetic flow logs via generate_mock_data.
        """
        if filepath and os.path.exists(filepath):
            logger.info(f"Loading raw flow logs from: {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df):,} flows with {len(df.columns)} columns.")
            return df

        if auto_generate_mock:
            target_path = filepath or "data/raw_flow_logs.csv"
            logger.warning(
                f"Dataset not found at '{filepath}'. Activating fallback generator to create "
                f"24 hours of synthetic network flow logs with attack surges..."
            )
            df = generate_mock_flows(
                duration_hours=24,
                output_path=target_path,
            )
            return df

        raise FileNotFoundError(f"Input file not found: {filepath}")

    def process(
        self,
        raw_df: pd.DataFrame,
        fill_empty_windows: bool = True,
    ) -> pd.DataFrame:
        """
        Process raw dataframe into aggregated time-series windows.

        Calculates per window:
        - total_flows: Count of flows.
        - attack_count: Malicious flows count.
        - risk_score: (attack_count / total_flows) * 100.
        - byte_volume: Sum of total fwd + bwd bytes.
        - distinct_src_ips: Unique count of source IP addresses.
        - syn_flag_count: Sum of SYN flag indicators.
        - failed_connections: Sum of failed connection attempts.

        Returns:
            pd.DataFrame with 'window_timestamp' and aggregated metrics.
        """
        if raw_df.empty:
            raise ValueError("Input dataframe is empty. Cannot aggregate windows.")

        # 1. Normalize schema
        norm_df = map_dataset_columns(raw_df)

        # 2. Parse timestamps
        norm_df["timestamp"] = parse_timestamps(norm_df["timestamp"])
        norm_df = norm_df.dropna(subset=["timestamp"])

        if norm_df.empty:
            raise ValueError("No valid timestamps could be parsed from input data.")

        # 3. Sort chronologically
        norm_df = norm_df.sort_values("timestamp").reset_index(drop=True)

        # 4. Group by fixed regularized time bins
        # Using pd.Grouper for consistent temporal binning
        grouper = pd.Grouper(key="timestamp", freq=self.window_size)
        agg_df = norm_df.groupby(grouper).agg(
            total_flows=("is_attack", "count"),
            attack_count=("is_attack", "sum"),
            byte_volume=("byte_volume", "sum"),
            distinct_src_ips=("src_ip", "nunique"),
            syn_flag_count=("syn_flag", "sum"),
            failed_connections=("failed_connection", "sum"),
        ).reset_index()

        agg_df.rename(columns={"timestamp": "window_timestamp"}, inplace=True)

        # 5. Regularize time grid: ensure complete continuous sequence with no missing intervals
        if fill_empty_windows and len(agg_df) > 0:
            start_ts = agg_df["window_timestamp"].min()
            end_ts = agg_df["window_timestamp"].max()
            full_grid = pd.date_range(
                start=start_ts,
                end=end_ts,
                freq=self.window_size,
                name="window_timestamp",
            )
            agg_df = agg_df.set_index("window_timestamp").reindex(full_grid).reset_index()

            # Fill missing windows with 0s
            zero_fill_cols = [
                "total_flows",
                "attack_count",
                "byte_volume",
                "distinct_src_ips",
                "syn_flag_count",
                "failed_connections",
            ]
            for col in zero_fill_cols:
                agg_df[col] = agg_df[col].fillna(0).astype(np.int64)

        # 6. Compute Risk Score (0 to 100 percentage)
        # Avoid division by zero: if total_flows is 0, risk_score is 0.0
        agg_df["risk_score"] = np.where(
            agg_df["total_flows"] > 0,
            np.round((agg_df["attack_count"] / agg_df["total_flows"]) * 100.0, 2),
            0.0,
        )

        # Ensure correct column ordering
        cols_order = [
            "window_timestamp",
            "total_flows",
            "attack_count",
            "risk_score",
            "byte_volume",
            "distinct_src_ips",
            "syn_flag_count",
            "failed_connections",
        ]
        agg_df = agg_df[cols_order]

        return agg_df

    def process_and_save(
        self,
        input_path: Optional[str] = "data/raw_flow_logs.csv",
        output_path: str = "data/windowed_traffic.csv",
        auto_generate_mock: bool = True,
    ) -> pd.DataFrame:
        """
        Execute full preprocessing pipeline and save results to CSV.
        """
        raw_df = self.load_data(input_path, auto_generate_mock=auto_generate_mock)
        agg_df = self.process(raw_df)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        agg_df.to_csv(output_path, index=False)
        logger.info(
            f"Successfully processed {len(raw_df):,} flows into {len(agg_df):,} regularized "
            f"time-series windows ({self.window_size}). Output saved to: {output_path}"
        )
        return agg_df


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and aggregate raw network flow logs into time-series windows."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_flow_logs.csv",
        help="Path to raw network flow CSV (CICIDS2017/UNSW-NB15/synthetic).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/windowed_traffic.csv",
        help="Path where aggregated windowed traffic CSV will be saved.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="5min",
        help="Time window aggregation bin size (e.g. 5min, 15min, 1h). Default: 5min",
    )
    parser.add_argument(
        "--no-auto-mock",
        action="store_true",
        help="Disable automatic synthetic data generation when input file is missing.",
    )

    args = parser.parse_args()

    preprocessor = FlowDataPreprocessor(window_size=args.window)
    preprocessor.process_and_save(
        input_path=args.input,
        output_path=args.output,
        auto_generate_mock=not args.no_auto_mock,
    )


if __name__ == "__main__":
    main()
