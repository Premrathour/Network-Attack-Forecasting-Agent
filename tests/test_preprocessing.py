"""
Unit Tests for Data Preprocessing Pipeline (Network Attack Forecasting Agent).

Verifies:
1. Column mapping and schema normalization (CICIDS2017, UNSW-NB15, Synthetic).
2. Timestamp parsing across ISO, Unix, and custom date formats.
3. Accurate 5-minute window aggregation and metric calculations (total_flows, attack_count,
   risk_score, byte_volume, distinct_src_ips, syn_flag_count, failed_connections).
4. Time grid regularizing and imputation of missing interval gaps.
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FlowDataPreprocessor,
    map_dataset_columns,
    normalize_column_name,
    parse_timestamps,
)
from src.generate_mock_data import generate_mock_flows


class TestColumnMapping:
    """Tests for dataset column normalization and schema unification."""

    def test_normalize_column_name(self):
        assert normalize_column_name(" Timestamp") == "timestamp"
        assert normalize_column_name("Total Length of Fwd Packets ") == "total_length_of_fwd_packets"
        assert normalize_column_name("SYN-Flag-Count!!") == "synflagcount"

    def test_map_dataset_columns_synthetic(self):
        raw = pd.DataFrame({
            "timestamp": ["2026-10-20 00:00:00"],
            "src_ip": ["192.168.1.10"],
            "dst_ip": ["10.0.0.5"],
            "fwd_bytes": [200],
            "bwd_bytes": [500],
            "syn_flag": [1],
            "failed_connection": [0],
            "label": ["BENIGN"],
        })
        mapped = map_dataset_columns(raw)
        assert "timestamp" in mapped.columns
        assert mapped["src_ip"].iloc[0] == "192.168.1.10"
        assert mapped["byte_volume"].iloc[0] == 700
        assert mapped["syn_flag"].iloc[0] == 1
        assert mapped["is_attack"].iloc[0] == 0

    def test_map_dataset_columns_cicids2017(self):
        raw = pd.DataFrame({
            " Timestamp": ["07/07/2017 8:30:00 AM"],
            " Source IP": ["192.168.10.50"],
            " Destination IP": ["172.16.0.1"],
            " Total Length of Fwd Packets": [1200],
            " Total Length of Bwd Packets": [3400],
            " SYN Flag Count": [1],
            " Label": ["DDoS"],
        })
        mapped = map_dataset_columns(raw)
        assert mapped["src_ip"].iloc[0] == "192.168.10.50"
        assert mapped["dst_ip"].iloc[0] == "172.16.0.1"
        assert mapped["fwd_bytes"].iloc[0] == 1200
        assert mapped["bwd_bytes"].iloc[0] == 3400
        assert mapped["byte_volume"].iloc[0] == 4600
        assert mapped["syn_flag"].iloc[0] == 1
        assert mapped["is_attack"].iloc[0] == 1
        assert mapped["failed_connection"].iloc[0] == 0

    def test_map_dataset_columns_unsw_nb15(self):
        raw = pd.DataFrame({
            "stime": [1421927414],
            "srcip": ["175.45.176.0"],
            "dstip": ["149.171.126.16"],
            "sbytes": [568],
            "dbytes": [312],
            "attack_cat": ["Exploits"],
            "label": [1],
        })
        mapped = map_dataset_columns(raw)
        assert mapped["src_ip"].iloc[0] == "175.45.176.0"
        assert mapped["byte_volume"].iloc[0] == 880
        assert mapped["is_attack"].iloc[0] == 1


class TestTimestampParsing:
    """Tests for datetime parsing across multiple standard formats."""

    def test_parse_iso_timestamps(self):
        ts = pd.Series(["2026-10-20 12:00:00", "2026-10-20 12:05:00"])
        parsed = parse_timestamps(ts)
        assert parsed.dt.hour.tolist() == [12, 12]
        assert parsed.dt.minute.tolist() == [0, 5]

    def test_parse_unix_epoch_timestamps(self):
        ts = pd.Series([1792454400, 1792454700])  # ~2026 epoch times
        parsed = parse_timestamps(ts)
        assert not parsed.isna().any()
        assert (parsed.iloc[1] - parsed.iloc[0]).total_seconds() == 300


class TestWindowAggregation:
    """Tests for window binning, metric aggregation, and grid regularization."""

    @pytest.fixture
    def sample_traffic_df(self):
        """
        Creates a controlled test set with known metrics:
        Window 1: 2026-10-20 10:00:00 to 10:04:59 (3 flows: 1 Benign, 2 Attack)
        Window 2: 2026-10-20 10:05:00 to 10:09:59 (1 flow: Benign)
        Window 3: 2026-10-20 10:10:00 to 10:14:59 (0 flows - Gap to be filled)
        Window 4: 2026-10-20 10:15:00 to 10:19:59 (2 flows: 2 Attack)
        """
        return pd.DataFrame([
            # Window 1 (10:00)
            {
                "timestamp": "2026-10-20 10:00:15",
                "src_ip": "192.168.1.10",
                "dst_ip": "10.0.0.5",
                "fwd_bytes": 100,
                "bwd_bytes": 200,
                "syn_flag": 0,
                "failed_connection": 0,
                "label": "BENIGN",
            },
            {
                "timestamp": "2026-10-20 10:01:30",
                "src_ip": "192.168.1.20",
                "dst_ip": "10.0.0.5",
                "fwd_bytes": 50,
                "bwd_bytes": 0,
                "syn_flag": 1,
                "failed_connection": 1,
                "label": "DDoS",
            },
            {
                "timestamp": "2026-10-20 10:03:00",
                "src_ip": "192.168.1.20",  # Duplicate IP in same window
                "dst_ip": "10.0.0.5",
                "fwd_bytes": 50,
                "bwd_bytes": 0,
                "syn_flag": 1,
                "failed_connection": 1,
                "label": "DDoS",
            },
            # Window 2 (10:05)
            {
                "timestamp": "2026-10-20 10:06:00",
                "src_ip": "192.168.1.30",
                "dst_ip": "10.0.0.8",
                "fwd_bytes": 300,
                "bwd_bytes": 700,
                "syn_flag": 0,
                "failed_connection": 0,
                "label": "BENIGN",
            },
            # Window 3 has no traffic (gap)
            # Window 4 (10:15)
            {
                "timestamp": "2026-10-20 10:15:20",
                "src_ip": "203.0.113.1",
                "dst_ip": "10.0.0.5",
                "fwd_bytes": 100,
                "bwd_bytes": 100,
                "syn_flag": 1,
                "failed_connection": 1,
                "label": "Brute Force",
            },
            {
                "timestamp": "2026-10-20 10:18:45",
                "src_ip": "203.0.113.2",
                "dst_ip": "10.0.0.5",
                "fwd_bytes": 100,
                "bwd_bytes": 100,
                "syn_flag": 1,
                "failed_connection": 1,
                "label": "Brute Force",
            },
        ])

    def test_window_calculations(self, sample_traffic_df):
        preprocessor = FlowDataPreprocessor(window_size="5min")
        agg = preprocessor.process(sample_traffic_df, fill_empty_windows=True)

        assert len(agg) == 4, f"Expected 4 regularized windows, got {len(agg)}"

        # Check Window 1 (10:00:00)
        w1 = agg.iloc[0]
        assert w1["total_flows"] == 3
        assert w1["attack_count"] == 2
        # risk_score = (2 / 3) * 100 = 66.67
        assert round(w1["risk_score"], 2) == 66.67
        # byte_volume = (100+200) + (50+0) + (50+0) = 400
        assert w1["byte_volume"] == 400
        # distinct IPs = {"192.168.1.10", "192.168.1.20"} -> 2
        assert w1["distinct_src_ips"] == 2
        assert w1["syn_flag_count"] == 2
        assert w1["failed_connections"] == 2

        # Check Window 2 (10:05:00)
        w2 = agg.iloc[1]
        assert w2["total_flows"] == 1
        assert w2["attack_count"] == 0
        assert w2["risk_score"] == 0.0
        assert w2["byte_volume"] == 1000
        assert w2["distinct_src_ips"] == 1
        assert w2["syn_flag_count"] == 0
        assert w2["failed_connections"] == 0

        # Check Window 3 (10:10:00) - Missing interval gap filled with zeros
        w3 = agg.iloc[2]
        assert w3["total_flows"] == 0
        assert w3["attack_count"] == 0
        assert w3["risk_score"] == 0.0
        assert w3["byte_volume"] == 0
        assert w3["distinct_src_ips"] == 0
        assert w3["syn_flag_count"] == 0
        assert w3["failed_connections"] == 0

        # Check Window 4 (10:15:00) - 100% attack traffic
        w4 = agg.iloc[3]
        assert w4["total_flows"] == 2
        assert w4["attack_count"] == 2
        assert w4["risk_score"] == 100.0
        assert w4["byte_volume"] == 400
        assert w4["distinct_src_ips"] == 2
        assert w4["syn_flag_count"] == 2
        assert w4["failed_connections"] == 2

    def test_empty_dataframe_handling(self):
        preprocessor = FlowDataPreprocessor(window_size="5min")
        with pytest.raises(ValueError):
            preprocessor.process(pd.DataFrame())

    def test_end_to_end_pipeline_with_mock_data(self):
        temp_dir = tempfile.mkdtemp()
        try:
            raw_path = os.path.join(temp_dir, "test_raw.csv")
            windowed_path = os.path.join(temp_dir, "test_windowed.csv")

            preprocessor = FlowDataPreprocessor(window_size="5min")
            # Should auto-generate 24h mock data
            agg = preprocessor.process_and_save(
                input_path=raw_path,
                output_path=windowed_path,
                auto_generate_mock=True,
            )

            assert os.path.exists(windowed_path)
            assert os.path.exists(raw_path)
            assert len(agg) == 288  # 24 hours * 12 (5-min intervals per hour) = 288 intervals

            # Check column names
            expected_cols = [
                "window_timestamp",
                "total_flows",
                "attack_count",
                "risk_score",
                "byte_volume",
                "distinct_src_ips",
                "syn_flag_count",
                "failed_connections",
            ]
            assert list(agg.columns) == expected_cols
            assert agg["total_flows"].sum() > 0
            assert agg["attack_count"].sum() > 0
            assert (agg["risk_score"] >= 0).all() and (agg["risk_score"] <= 100).all()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
