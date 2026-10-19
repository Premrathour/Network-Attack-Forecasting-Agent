"""
Feature Engineering and Dataset Splitting Module for Network Attack Forecasting Agent.
(SIH 2026 - SIH26153)

Loads regularized windowed traffic (5-min windows) and builds:
1. Temporal features:
   - Lagged targets: Lags for attack_count and risk_score at t-1, t-2, t-3, t-6, t-12.
   - Rolling stats: Rolling mean and std over 3, 6, and 12 periods.
   - Cyclical temporal indicators: Sin/Cos encoding for time-of-day and day-of-week.
2. Chronological dataset splitting (70% train, 15% validation, 15% test; no shuffling).
3. Dual output formatting:
   - Tabular arrays (X, y) for gradient boosting / tree-based models.
   - 3D sequential sliding tensors (batch_size, sequence_length=12, feature_dim)
     predicting targets at t+1 and t+3 for deep learning sequence models.
4. Artifact serialization into data/processed/.
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("features")


def add_cyclical_temporal_features(
    df: pd.DataFrame,
    timestamp_col: str = "window_timestamp",
) -> pd.DataFrame:
    """
    Encode time-of-day and day-of-week as smooth cyclical sine and cosine signals.
    Minute of day: [0, 1440) -> 2 * pi * minute / 1440
    Day of week: [0, 7) -> 2 * pi * day / 7
    """
    res = df.copy()
    ts = pd.to_datetime(res[timestamp_col])

    # Minute of day in [0, 1440)
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    res["time_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    res["time_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

    # Day of week in [0, 7)
    day_of_week = ts.dt.dayofweek
    res["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    res["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    return res


def add_lagged_features(
    df: pd.DataFrame,
    target_cols: List[str] = ["attack_count", "risk_score"],
    lags: List[int] = [1, 2, 3, 6, 12],
) -> pd.DataFrame:
    """
    Generate historical lag features for specified target metrics.
    Ensures strictly causal historical observations (no future data leakage).
    """
    res = df.copy()
    for col in target_cols:
        if col not in res.columns:
            logger.warning(f"Target column '{col}' not found for lag feature generation.")
            continue
        for lag in lags:
            res[f"{col}_lag_{lag}"] = res[col].shift(lag)
    return res


def add_rolling_features(
    df: pd.DataFrame,
    target_cols: List[str] = ["attack_count", "risk_score"],
    windows: List[int] = [3, 6, 12],
) -> pd.DataFrame:
    """
    Compute rolling mean and rolling standard deviation over specified historical windows.
    Always uses .shift(1) before rolling to guarantee current window observation is NOT leaked.
    """
    res = df.copy()
    for col in target_cols:
        if col not in res.columns:
            logger.warning(f"Target column '{col}' not found for rolling feature generation.")
            continue
        shifted = res[col].shift(1)
        for w in windows:
            res[f"{col}_roll_mean_{w}"] = shifted.rolling(window=w).mean()
            res[f"{col}_roll_std_{w}"] = shifted.rolling(window=w).std().fillna(0.0)
    return res


def create_sequential_sliding_tensors(
    feature_matrix: np.ndarray,
    target_matrix: np.ndarray,
    sequence_length: int = 12,
    target_horizons: List[int] = [1, 3],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct 3D sequential sliding tensors for deep learning sequence models (LSTM/GRU/Transformer).

    Args:
        feature_matrix: 2D array of features of shape (T, num_features).
        target_matrix: 1D or 2D array of target variable(s) of shape (T,) or (T, num_targets).
        sequence_length: Number of past timesteps in input sequence window (default: 12).
        target_horizons: Horizons into the future to predict (default: [1, 3] -> t+1 and t+3).

    Returns:
        X_seq: 3D tensor of shape (num_samples, sequence_length, num_features).
        y_seq: 2D array of shape (num_samples, len(target_horizons)).
    """
    if target_matrix.ndim == 1:
        target_matrix = target_matrix.reshape(-1, 1)

    max_horizon = max(target_horizons)
    total_timesteps = len(feature_matrix)

    # Number of valid sliding windows
    num_samples = total_timesteps - sequence_length - max_horizon + 1
    if num_samples <= 0:
        raise ValueError(
            f"Not enough timesteps ({total_timesteps}) for sequence_length={sequence_length} "
            f"and max_horizon={max_horizon}."
        )

    num_features = feature_matrix.shape[1]
    num_targets = len(target_horizons)

    X_seq = np.zeros((num_samples, sequence_length, num_features), dtype=np.float32)
    y_seq = np.zeros((num_samples, num_targets), dtype=np.float32)

    for i in range(num_samples):
        # Input sequence ends at time step i + sequence_length - 1
        end_idx = i + sequence_length
        X_seq[i] = feature_matrix[i:end_idx]

        # Targets at t + horizon
        # For horizon h=1, target index is end_idx (i.e. timestamp right after the sequence)
        # For horizon h=3, target index is end_idx + 2
        for h_idx, h in enumerate(target_horizons):
            target_idx = end_idx + (h - 1)
            y_seq[i, h_idx] = target_matrix[target_idx, 0]

    return X_seq, y_seq


def chronological_split(
    data_length: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Calculate non-shuffled, strictly forward-temporal split slice bounds.
    70% train, 15% validation, 15% test.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    train_end = int(np.floor(data_length * train_ratio))
    val_end = int(np.floor(data_length * (train_ratio + val_ratio)))

    train_slice = (0, train_end)
    val_slice = (train_end, val_end)
    test_slice = (val_end, data_length)

    return train_slice, val_slice, test_slice


class TimeSeriesFeatureEngineer:
    """
    End-to-end feature pipeline for Network Attack Forecasting:
    - Generates temporal, cyclical, lag, and rolling features.
    - Performs chronological train/val/test splits without leakage.
    - Scales features using train-set statistics.
    - Exports tabular arrays and 3D sliding sequence tensors.
    """

    def __init__(
        self,
        lags: List[int] = [1, 2, 3, 6, 12],
        rolling_windows: List[int] = [3, 6, 12],
        sequence_length: int = 12,
        target_horizons: List[int] = [1, 2, 3],
        primary_target: str = "attack_count",
        secondary_target: str = "risk_score",
    ):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.sequence_length = sequence_length
        self.target_horizons = target_horizons
        self.primary_target = primary_target
        self.secondary_target = secondary_target

        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.raw_feature_columns: List[str] = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full feature engineering suite to windowed dataframe.
        """
        logger.info("Generating cyclical, lag, and rolling features...")
        feat_df = df.copy()

        # 1. Cyclical temporal indicators
        feat_df = add_cyclical_temporal_features(feat_df, timestamp_col="window_timestamp")

        # 2. Historical lag features
        feat_df = add_lagged_features(
            feat_df,
            target_cols=[self.primary_target, self.secondary_target],
            lags=self.lags,
        )

        # Also add lags for traffic context
        traffic_context_cols = ["total_flows", "byte_volume", "syn_flag_count", "failed_connections"]
        for c in traffic_context_cols:
            if c in feat_df.columns:
                for l in [1, 2, 3]:
                    feat_df[f"{c}_lag_{l}"] = feat_df[c].shift(l)

        # 3. Rolling window statistics
        feat_df = add_rolling_features(
            feat_df,
            target_cols=[self.primary_target, self.secondary_target],
            windows=self.rolling_windows,
        )

        # 4. Target multi-horizon forward values for tabular gradient boosting models
        for h in self.target_horizons:
            feat_df[f"{self.primary_target}_target_t{h}"] = feat_df[self.primary_target].shift(-h)
            feat_df[f"{self.secondary_target}_target_t{h}"] = feat_df[self.secondary_target].shift(-h)

        return feat_df

    def process_and_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, Union[np.ndarray, List[str], pd.DataFrame]]:
        """
        Execute feature engineering, clean NaNs from boundary shifts, perform chronological
        splitting, normalize features, and build both tabular and 3D sequence formats.
        """
        engineered_df = self.engineer_features(df)

        # Identify all feature columns (exclude raw timestamp and targets)
        target_cols = [
            f"{self.primary_target}_target_t{h}" for h in self.target_horizons
        ] + [f"{self.secondary_target}_target_t{h}" for h in self.target_horizons]

        exclude_cols = set(["window_timestamp"] + target_cols)
        self.feature_columns = [c for c in engineered_df.columns if c not in exclude_cols]

        # Clean NaN rows resulting from max lag (12) and max forward shift (3)
        valid_df = engineered_df.dropna(subset=self.feature_columns + target_cols).reset_index(drop=True)
        logger.info(
            f"Engineered {len(self.feature_columns)} features across {len(valid_df)} valid temporal windows."
        )

        # -------------------------------------------------------------
        # Chronological Split (70% train, 15% val, 15% test)
        # -------------------------------------------------------------
        n = len(valid_df)
        train_slice, val_slice, test_slice = chronological_split(
            n, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
        )

        train_df = valid_df.iloc[train_slice[0]:train_slice[1]].copy()
        val_df = valid_df.iloc[val_slice[0]:val_slice[1]].copy()
        test_df = valid_df.iloc[test_slice[0]:test_slice[1]].copy()

        logger.info(
            f"Chronological Split: Train={len(train_df)} ({train_slice[0]}:{train_slice[1]}), "
            f"Val={len(val_df)} ({val_slice[0]}:{val_slice[1]}), "
            f"Test={len(test_df)} ({test_slice[0]}:{test_slice[1]})"
        )

        # -------------------------------------------------------------
        # 1. Feature Normalization (Scaler fitted STRICTLY on train)
        # -------------------------------------------------------------
        X_all = valid_df[self.feature_columns].values
        self.scaler.fit(X_all[:train_slice[1]])  # Fit only on train partition

        X_all_scaled = self.scaler.transform(X_all)
        target_cols_tab = [f"{self.primary_target}_target_t{h}" for h in self.target_horizons]
        y_all = valid_df[target_cols_tab].values

        # Tabular partitions
        X_train_tab = X_all[train_slice[0]:train_slice[1]]
        X_train_tab_scaled = X_all_scaled[train_slice[0]:train_slice[1]]
        y_train_tab = y_all[train_slice[0]:train_slice[1]]

        X_val_tab = X_all[val_slice[0]:val_slice[1]]
        X_val_tab_scaled = X_all_scaled[val_slice[0]:val_slice[1]]
        y_val_tab = y_all[val_slice[0]:val_slice[1]]

        X_test_tab = X_all[test_slice[0]:test_slice[1]]
        X_test_tab_scaled = X_all_scaled[test_slice[0]:test_slice[1]]
        y_test_tab = y_all[test_slice[0]:test_slice[1]]

        # -------------------------------------------------------------
        # 2. 3D Sequential Sliding Tensors for Deep Learning Models
        # -------------------------------------------------------------
        # Build sliding sequence tensors across continuous timeline
        # Target matrix uses current values of primary target, and sliding tensor
        # extractor looks ahead by target_horizons (t+1, t+3)
        raw_target_series = valid_df[[self.primary_target]].values
        X_seq_all, y_seq_all = create_sequential_sliding_tensors(
            X_all_scaled,
            raw_target_series,
            sequence_length=self.sequence_length,
            target_horizons=self.target_horizons,
        )

        # Chronologically partition the 3D sliding sequence windows
        seq_n = len(X_seq_all)
        tr_seq_slice, val_seq_slice, te_seq_slice = chronological_split(
            seq_n, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
        )

        X_train_seq = X_seq_all[tr_seq_slice[0]:tr_seq_slice[1]]
        y_train_seq = y_seq_all[tr_seq_slice[0]:tr_seq_slice[1]]

        X_val_seq = X_seq_all[val_seq_slice[0]:val_seq_slice[1]]
        y_val_seq = y_seq_all[val_seq_slice[0]:val_seq_slice[1]]

        X_test_seq = X_seq_all[te_seq_slice[0]:te_seq_slice[1]]
        y_test_seq = y_seq_all[te_seq_slice[0]:te_seq_slice[1]]

        logger.info(
            f"3D Sequential Tensors Created -> "
            f"Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}"
        )

        return {
            "feature_columns": self.feature_columns,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "X_train_tab": X_train_tab,
            "X_train_tab_scaled": X_train_tab_scaled,
            "y_train_tab": y_train_tab,
            "X_val_tab": X_val_tab,
            "X_val_tab_scaled": X_val_tab_scaled,
            "y_val_tab": y_val_tab,
            "X_test_tab": X_test_tab,
            "X_test_tab_scaled": X_test_tab_scaled,
            "y_test_tab": y_test_tab,
            "X_train_seq": X_train_seq,
            "y_train_seq": y_train_seq,
            "X_val_seq": X_val_seq,
            "y_val_seq": y_val_seq,
            "X_test_seq": X_test_seq,
            "y_test_seq": y_test_seq,
        }

    def save_artifacts(
        self,
        artifacts: Dict[str, Union[np.ndarray, List[str], pd.DataFrame]],
        output_dir: str = "data/processed",
    ) -> None:
        """
        Serialize tabular arrays, 3D sequence tensors, metadata, and dataframes to disk.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save Tabular Arrays (.npz)
        tabular_npz_path = os.path.join(output_dir, "tabular_features.npz")
        np.savez_compressed(
            tabular_npz_path,
            X_train=artifacts["X_train_tab_scaled"],
            y_train=artifacts["y_train_tab"],
            X_val=artifacts["X_val_tab_scaled"],
            y_val=artifacts["y_val_tab"],
            X_test=artifacts["X_test_tab_scaled"],
            y_test=artifacts["y_test_tab"],
        )
        logger.info(f"Saved Tabular dataset array to: {tabular_npz_path}")

        # 2. Save 3D Sequential Tensors (.npz)
        sequence_npz_path = os.path.join(output_dir, "sequence_tensors.npz")
        np.savez_compressed(
            sequence_npz_path,
            X_train=artifacts["X_train_seq"],
            y_train=artifacts["y_train_seq"],
            X_val=artifacts["X_val_seq"],
            y_val=artifacts["y_val_seq"],
            X_test=artifacts["X_test_seq"],
            y_test=artifacts["y_test_seq"],
        )
        logger.info(f"Saved 3D Sequential tensors to: {sequence_npz_path}")

        # 3. Save Tabular CSVs for easy inspection
        artifacts["train_df"].to_csv(os.path.join(output_dir, "train_tabular.csv"), index=False)
        artifacts["val_df"].to_csv(os.path.join(output_dir, "val_tabular.csv"), index=False)
        artifacts["test_df"].to_csv(os.path.join(output_dir, "test_tabular.csv"), index=False)

        # 4. Save Feature Metadata JSON
        metadata = {
            "feature_columns": artifacts["feature_columns"],
            "num_features": len(artifacts["feature_columns"]),
            "sequence_length": self.sequence_length,
            "target_horizons": self.target_horizons,
            "target_names": [f"{self.primary_target}_t{h}" for h in self.target_horizons],
            "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "shapes": {
                "tabular": {
                    "X_train": list(artifacts["X_train_tab_scaled"].shape),
                    "y_train": list(artifacts["y_train_tab"].shape),
                    "X_val": list(artifacts["X_val_tab_scaled"].shape),
                    "y_val": list(artifacts["y_val_tab"].shape),
                    "X_test": list(artifacts["X_test_tab_scaled"].shape),
                    "y_test": list(artifacts["y_test_tab"].shape),
                },
                "sequential_3d": {
                    "X_train": list(artifacts["X_train_seq"].shape),
                    "y_train": list(artifacts["y_train_seq"].shape),
                    "X_val": list(artifacts["X_val_seq"].shape),
                    "y_val": list(artifacts["y_val_seq"].shape),
                    "X_test": list(artifacts["X_test_seq"].shape),
                    "y_test": list(artifacts["y_test_seq"].shape),
                },
            },
        }

        metadata_path = os.path.join(output_dir, "feature_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved Feature Metadata to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run feature engineering and chronological dataset splitting."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/windowed_traffic.csv",
        help="Path to windowed traffic CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed feature tensors and metadata",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=12,
        help="Sequence length for 3D tensors (default: 12)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Input file not found at {args.input}. Please run src/preprocessing.py first."
        )

    logger.info(f"Loading windowed traffic from: {args.input}")
    df = pd.read_csv(args.input)

    engineer = TimeSeriesFeatureEngineer(sequence_length=args.seq_len)
    artifacts = engineer.process_and_split(df)
    engineer.save_artifacts(artifacts, output_dir=args.output_dir)
    print("\n[SUCCESS] Feature engineering and dataset splitting completed successfully!")


if __name__ == "__main__":
    main()
