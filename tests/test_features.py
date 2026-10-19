"""
Unit Tests for Feature Engineering & Dataset Splitting Module (SIH 2026 - SIH26153).

Verifies:
1. Cyclical sine/cosine temporal transformations.
2. Lagged target generation with zero lookahead leakage.
3. Rolling statistics (mean, std) calculated strictly over historical windows.
4. Chronological dataset splitting (70% train, 15% val, 15% test).
5. 3D sequential sliding tensors construction for deep learning sequence models.
6. Artifact persistence and metadata serialization in data/processed/.
"""

import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.features import (
    TimeSeriesFeatureEngineer,
    add_cyclical_temporal_features,
    add_lagged_features,
    add_rolling_features,
    chronological_split,
    create_sequential_sliding_tensors,
)


class TestTemporalAndCyclicalFeatures:
    """Tests for time-based transformations and cyclical sin/cos encodings."""

    def test_cyclical_temporal_bounds_and_periodicity(self):
        df = pd.DataFrame({
            "window_timestamp": [
                "2026-10-20 00:00:00",  # Midnight (0 min)
                "2026-10-20 06:00:00",  # 6 AM (360 min) -> pi/2
                "2026-10-20 12:00:00",  # Noon (720 min) -> pi
                "2026-10-20 18:00:00",  # 6 PM (1080 min) -> 3pi/2
            ]
        })
        res = add_cyclical_temporal_features(df)

        assert "time_of_day_sin" in res.columns
        assert "time_of_day_cos" in res.columns
        assert "day_of_week_sin" in res.columns
        assert "day_of_week_cos" in res.columns

        # Check bounds
        assert (res["time_of_day_sin"] >= -1.0).all() and (res["time_of_day_sin"] <= 1.0).all()
        assert (res["time_of_day_cos"] >= -1.0).all() and (res["time_of_day_cos"] <= 1.0).all()

        # Midnight: sin(0) = 0, cos(0) = 1
        np.testing.assert_almost_equal(res["time_of_day_sin"].iloc[0], 0.0, decimal=4)
        np.testing.assert_almost_equal(res["time_of_day_cos"].iloc[0], 1.0, decimal=4)

        # Noon: sin(pi) = 0, cos(pi) = -1
        np.testing.assert_almost_equal(res["time_of_day_sin"].iloc[2], 0.0, decimal=4)
        np.testing.assert_almost_equal(res["time_of_day_cos"].iloc[2], -1.0, decimal=4)


class TestLagsAndRollingFeatures:
    """Tests for lag features and rolling window stats with strict causality."""

    def test_lagged_features_values(self):
        df = pd.DataFrame({
            "attack_count": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
            "risk_score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        })
        lags = [1, 2, 3, 6, 12]
        res = add_lagged_features(df, target_cols=["attack_count", "risk_score"], lags=lags)

        # Index 13 (attack_count=140):
        # lag_1 = 130, lag_2 = 120, lag_3 = 110, lag_6 = 80, lag_12 = 20
        idx = 13
        assert res.loc[idx, "attack_count_lag_1"] == 130
        assert res.loc[idx, "attack_count_lag_2"] == 120
        assert res.loc[idx, "attack_count_lag_3"] == 110
        assert res.loc[idx, "attack_count_lag_6"] == 80
        assert res.loc[idx, "attack_count_lag_12"] == 20

        # Verify that lag_12 has 12 NaNs at start
        assert res["attack_count_lag_12"].iloc[:12].isna().all()
        assert not res["attack_count_lag_12"].iloc[12:].isna().any()

    def test_rolling_features_no_current_leakage(self):
        # Rolling windows: 3, 6, 12
        values = list(range(1, 20))
        df = pd.DataFrame({
            "attack_count": values,
            "risk_score": [v * 0.5 for v in values],
        })
        res = add_rolling_features(df, target_cols=["attack_count"], windows=[3])

        # At index 4 (value = 5):
        # Shifted historical values are [1, 2, 3, 4]
        # Rolling 3 mean over shifted values at index 4 is mean([2, 3, 4]) = 3.0
        # If current value (5) leaked, mean would be (3+4+5)/3 = 4.0
        assert res.loc[4, "attack_count_roll_mean_3"] == 3.0


class TestSplittingAndSequenceTensors:
    """Tests for chronological splitting and 3D sliding sequence generation."""

    def test_chronological_split_proportions(self):
        total_len = 100
        train_slice, val_slice, test_slice = chronological_split(
            total_len, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
        )

        assert train_slice == (0, 70)
        assert val_slice == (70, 85)
        assert test_slice == (85, 100)

        # Ensure no overlap and complete coverage
        train_idx = set(range(*train_slice))
        val_idx = set(range(*val_slice))
        test_idx = set(range(*test_slice))

        assert len(train_idx.intersection(val_idx)) == 0
        assert len(val_idx.intersection(test_idx)) == 0
        assert len(train_idx.union(val_idx).union(test_idx)) == total_len

    def test_sequential_sliding_tensors_shape_and_alignment(self):
        # 30 timesteps, 5 features
        num_timesteps = 30
        num_features = 5
        features = np.arange(num_timesteps * num_features, dtype=np.float32).reshape(num_timesteps, num_features)
        # Target: single variable = 100 + timestep
        targets = (100 + np.arange(num_timesteps, dtype=np.float32)).reshape(-1, 1)

        seq_len = 12
        horizons = [1, 3]  # t+1 and t+3
        max_h = 3

        X_seq, y_seq = create_sequential_sliding_tensors(
            features, targets, sequence_length=seq_len, target_horizons=horizons
        )

        expected_samples = num_timesteps - seq_len - max_h + 1  # 30 - 12 - 3 + 1 = 16
        assert X_seq.shape == (expected_samples, seq_len, num_features)
        assert y_seq.shape == (expected_samples, 2)

        # Check sample 0:
        # Sequence input: timesteps 0 to 11
        np.testing.assert_array_equal(X_seq[0], features[0:12])
        # Target t+1: timestep 12 -> targets[12] = 112
        assert y_seq[0, 0] == 112.0
        # Target t+3: timestep 14 -> targets[14] = 114
        assert y_seq[0, 1] == 114.0


class TestPipelineEndToEnd:
    """Tests end-to-end execution of TimeSeriesFeatureEngineer."""

    def test_feature_engineer_end_to_end_artifacts(self):
        temp_dir = tempfile.mkdtemp()
        try:
            # Generate a 100-window dummy dataframe
            dates = pd.date_range("2026-10-19 00:00:00", periods=100, freq="5min")
            dummy_df = pd.DataFrame({
                "window_timestamp": dates,
                "total_flows": np.random.randint(50, 500, size=100),
                "attack_count": np.random.randint(0, 100, size=100),
                "risk_score": np.random.uniform(0.0, 50.0, size=100),
                "byte_volume": np.random.randint(10000, 500000, size=100),
                "distinct_src_ips": np.random.randint(10, 80, size=100),
                "syn_flag_count": np.random.randint(0, 50, size=100),
                "failed_connections": np.random.randint(0, 30, size=100),
            })

            engineer = TimeSeriesFeatureEngineer(
                lags=[1, 2, 3, 6, 12],
                rolling_windows=[3, 6, 12],
                sequence_length=12,
                target_horizons=[1, 3],
            )

            artifacts = engineer.process_and_split(dummy_df)
            engineer.save_artifacts(artifacts, output_dir=temp_dir)

            # Check that files were created
            assert os.path.exists(os.path.join(temp_dir, "tabular_features.npz"))
            assert os.path.exists(os.path.join(temp_dir, "sequence_tensors.npz"))
            assert os.path.exists(os.path.join(temp_dir, "feature_metadata.json"))
            assert os.path.exists(os.path.join(temp_dir, "train_tabular.csv"))
            assert os.path.exists(os.path.join(temp_dir, "val_tabular.csv"))
            assert os.path.exists(os.path.join(temp_dir, "test_tabular.csv"))

            # Verify contents of tabular_features.npz
            tab_npz = np.load(os.path.join(temp_dir, "tabular_features.npz"))
            assert "X_train" in tab_npz and "y_train" in tab_npz
            assert "X_val" in tab_npz and "y_val" in tab_npz
            assert "X_test" in tab_npz and "y_test" in tab_npz

            # Verify contents of sequence_tensors.npz
            seq_npz = np.load(os.path.join(temp_dir, "sequence_tensors.npz"))
            assert seq_npz["X_train"].ndim == 3
            assert seq_npz["X_train"].shape[1] == 12  # sequence length
            assert seq_npz["y_train"].shape[1] == 2   # horizons t+1, t+3

            # Verify feature metadata json
            with open(os.path.join(temp_dir, "feature_metadata.json")) as f:
                meta = json.load(f)
            assert meta["sequence_length"] == 12
            assert meta["target_horizons"] == [1, 3]
            assert "attack_count_lag_1" in meta["feature_columns"]
            assert "time_of_day_sin" in meta["feature_columns"]
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
