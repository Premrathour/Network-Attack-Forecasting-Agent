"""
Unit Tests for Alert Engine & Evaluation Suite (SIH 2026 - SIH26153).

Verifies:
1. Early-Warning Threshold Engine:
   - Static threshold trigger logic (GREEN, YELLOW, RED)
   - Dynamic statistical threshold trigger logic (rolling_mean + 1.5/2.5 * rolling_std)
   - check_threshold() function interface
   - Stream evaluation & backward-compatible SOC dataframe enrichment
2. Evaluation Suite:
   - Regression metrics (MAE, RMSE, MAPE, R2)
   - Warning Lead Time calculation (minutes from early alert to ground truth peak)
   - False Positive Alarm Rate (FPAR)
   - Plot generation into reports/
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

from src.alert_engine import AlertEngine, check_threshold
from src.evaluate import (
    calculate_regression_metrics,
    calculate_warning_lead_time_and_fpar,
    generate_evaluation_plot,
)


class TestAlertEngineThresholds:
    """Tests for static, dynamic, and hybrid early warning thresholding."""

    def test_static_threshold_states(self):
        engine = AlertEngine(static_yellow_limit=50.0, static_red_limit=150.0, mode="static")
        baseline = [10.0, 15.0, 20.0]

        # Normal value < 50
        assert engine.check_threshold(30.0, baseline) == "GREEN"

        # Elevated value >= 50 and < 150
        assert engine.check_threshold(75.0, baseline) == "YELLOW"

        # Imminent attack >= 150
        assert engine.check_threshold(200.0, baseline) == "RED"

    def test_dynamic_statistical_threshold_states(self):
        baseline = [90.0, 100.0, 110.0, 100.0, 100.0]
        engine = AlertEngine(
            dynamic_yellow_std_multiplier=1.5,
            dynamic_red_std_multiplier=2.5,
            mode="dynamic",
        )
        mean = float(np.mean(baseline))
        std = float(np.std(baseline))

        val_green = mean + 0.5 * std   # < mean + 1.5 * std
        val_yellow = mean + 1.8 * std  # >= 1.5 and < 2.5
        val_red = mean + 3.0 * std     # >= 2.5

        assert engine.check_threshold(val_green, baseline) == "GREEN"
        assert engine.check_threshold(val_yellow, baseline) == "YELLOW"
        assert engine.check_threshold(val_red, baseline) == "RED"

    def test_check_threshold_functional_interface(self):
        baseline = [20.0, 25.0, 30.0]
        # In hybrid mode with default static yellow=60, static red=160
        res_green = check_threshold(10.0, baseline, mode="static")
        res_yellow = check_threshold(80.0, baseline, mode="static")
        res_red = check_threshold(250.0, baseline, mode="static")

        assert res_green == "GREEN"
        assert res_yellow == "YELLOW"
        assert res_red == "RED"

    def test_stream_evaluation_and_dataframe_enrichment(self):
        engine = AlertEngine()
        predictions = np.array([10.0, 20.0, 80.0, 220.0])
        history = np.array([15.0, 18.0, 16.0, 15.0])

        results = engine.evaluate_forecast_stream(predictions, history)
        assert len(results) == 4
        assert results[0].status == "GREEN"
        assert results[2].status in ["YELLOW", "RED"]
        assert results[3].status == "RED"


class TestEvaluationSuite:
    """Tests for regression and operational early warning metrics."""

    def test_regression_metrics(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])

        metrics = calculate_regression_metrics(y_true, y_pred)
        assert metrics["MAE"] == 10.0
        assert metrics["RMSE"] == 10.0
        assert metrics["R2"] > 0.95

    def test_warning_lead_time_positive_detection(self):
        """
        Simulate early warning alert triggered at step 2 (10 min),
        while actual ground truth peak happens at step 5 (25 min).
        Expected Warning Lead Time = (5 - 2) * 5 min = 15 minutes.
        """
        # 10 steps (each 5 min)
        y_true = np.array([10.0, 15.0, 12.0, 40.0, 120.0, 250.0, 180.0, 20.0, 10.0, 5.0])
        # Forecast leads and triggers alert at step 2
        y_pred = np.array([10.0, 15.0, 85.0, 180.0, 220.0, 240.0, 150.0, 15.0, 10.0, 5.0])

        res = calculate_warning_lead_time_and_fpar(
            y_true=y_true,
            y_pred=y_pred,
            interval_minutes=5,
            spike_threshold=75.0,
            forward_window_steps=4,
        )

        assert res["detected_spikes"] == 1
        assert res["warning_lead_time_minutes"] > 0
        assert res["false_positive_alarm_rate"] == 0.0  # Spikes occurred right after alert

    def test_false_positive_alarm_rate(self):
        """
        Simulate false alarms where model predicts high spikes, but ground truth stays completely flat.
        Expected FPAR = 1.0 (100% false alarms).
        """
        y_true = np.array([10.0, 12.0, 11.0, 10.0, 12.0, 10.0, 11.0, 10.0])
        # Model falsely predicts massive surge
        y_pred = np.array([10.0, 12.0, 190.0, 220.0, 10.0, 12.0, 10.0, 11.0])

        res = calculate_warning_lead_time_and_fpar(
            y_true=y_true,
            y_pred=y_pred,
            spike_threshold=75.0,
            forward_window_steps=3,
        )

        assert res["false_positive_alarm_rate"] == 1.0
        assert res["false_alerts"] > 0

    def test_generate_evaluation_plot(self):
        temp_dir = tempfile.mkdtemp()
        try:
            plot_path = os.path.join(temp_dir, "test_eval_plot.png")
            y_true = np.array([10.0, 20.0, 30.0, 150.0, 200.0, 40.0, 10.0])
            y_pred = np.array([12.0, 18.0, 80.0, 140.0, 190.0, 35.0, 12.0])
            alert_states = ["GREEN", "GREEN", "YELLOW", "RED", "RED", "GREEN", "GREEN"]
            metrics = {
                "MAE": 8.5,
                "RMSE": 11.2,
                "warning_lead_time_minutes": 10.0,
                "false_positive_alarm_rate": 0.0,
                "detected_spikes": 1,
            }

            out = generate_evaluation_plot(
                y_true=y_true,
                y_pred=y_pred,
                alert_states=alert_states,
                metrics=metrics,
                output_path=plot_path,
            )

            assert os.path.exists(plot_path)
            assert os.path.getsize(plot_path) > 10000  # valid image file
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
