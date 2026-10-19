"""
Unit Tests for Machine Learning & Deep Learning Models (SIH 2026 - SIH26153).

Verifies:
1. Baseline Multi-Horizon XGBoost Regressor (predictions at t+1 and t+3).
2. PyTorch 2-layer LSTM Recurrent Neural Network (architecture, multi-step output, checkpointing).
3. Early stopping and weight serialization.
4. Unified train_and_save() entrypoint.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from src.models import (
    AttackLSTMNetwork,
    DeepLearningForecaster,
    XGBoostMultiHorizonForecaster,
    train_and_save,
)


class TestBaselineXGBoost:
    """Tests for multi-horizon XGBoost baseline regressor."""

    @pytest.fixture
    def dummy_tabular_data(self):
        np.random.seed(42)
        N_train, N_val, N_test = 60, 15, 15
        num_features = 20

        X_train = np.random.randn(N_train, num_features).astype(np.float32)
        y_train = np.random.randint(0, 100, size=(N_train, 2)).astype(np.float32)

        X_val = np.random.randn(N_val, num_features).astype(np.float32)
        y_val = np.random.randint(0, 100, size=(N_val, 2)).astype(np.float32)

        X_test = np.random.randn(N_test, num_features).astype(np.float32)
        y_test = np.random.randint(0, 100, size=(N_test, 2)).astype(np.float32)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def test_xgboost_fit_predict_and_evaluate(self, dummy_tabular_data):
        X_train, y_train, X_val, y_val, X_test, y_test = dummy_tabular_data

        model = XGBoostMultiHorizonForecaster(n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        assert preds.shape == (len(X_test), 2)
        assert (preds >= 0).all(), "Predictions should be non-negative attack counts"

        metrics = model.evaluate(X_test, y_test)
        assert "t+1" in metrics and "t+3" in metrics
        assert "MAE" in metrics["t+1"] and "RMSE" in metrics["t+1"] and "R2" in metrics["t+1"]

    def test_xgboost_save_heads(self, dummy_tabular_data):
        X_train, y_train, _, _, _, _ = dummy_tabular_data
        temp_dir = tempfile.mkdtemp()
        try:
            model = XGBoostMultiHorizonForecaster(n_estimators=5, max_depth=2)
            model.fit(X_train, y_train)

            save_prefix = os.path.join(temp_dir, "test_xgb")
            model.save(save_prefix)

            assert os.path.exists(f"{save_prefix}_head_0.json")
            assert os.path.exists(f"{save_prefix}_head_1.json")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestDeepLearningLSTM:
    """Tests for PyTorch 2-layer LSTM Recurrent Neural Network."""

    def test_network_architecture_and_forward_pass(self):
        batch_size = 8
        seq_len = 12
        input_dim = 25
        output_dim = 3

        net = AttackLSTMNetwork(
            input_dim=input_dim,
            hidden_dim=32,
            num_layers=2,
            dropout=0.2,
            output_dim=output_dim,
            cell_type="lstm",
        )

        assert net.rnn.num_layers == 2
        assert net.fc.out_features == 3

        dummy_input = torch.randn(batch_size, seq_len, input_dim)
        output = net(dummy_input)

        assert output.shape == (batch_size, output_dim)

    def test_forecaster_fit_early_stopping_and_persistence(self):
        temp_dir = tempfile.mkdtemp()
        try:
            np.random.seed(42)
            torch.manual_seed(42)
            N_train, N_val, N_test = 40, 10, 10
            seq_len = 12
            input_dim = 15
            output_dim = 3

            X_train = np.random.randn(N_train, seq_len, input_dim).astype(np.float32)
            y_train = np.random.randint(0, 80, size=(N_train, output_dim)).astype(np.float32)

            X_val = np.random.randn(N_val, seq_len, input_dim).astype(np.float32)
            y_val = np.random.randint(0, 80, size=(N_val, output_dim)).astype(np.float32)

            X_test = np.random.randn(N_test, seq_len, input_dim).astype(np.float32)
            y_test = np.random.randint(0, 80, size=(N_test, output_dim)).astype(np.float32)

            forecaster = DeepLearningForecaster(
                input_dim=input_dim,
                hidden_dim=32,
                num_layers=2,
                dropout=0.2,
                output_dim=output_dim,
                cell_type="lstm",
                learning_rate=0.01,
            )

            # Fit with small epochs
            forecaster.fit(
                X_train, y_train, X_val, y_val, epochs=5, batch_size=8, patience=3
            )

            assert 1 <= len(forecaster.training_history["train_loss"]) <= 5
            assert 1 <= len(forecaster.training_history["val_loss"]) <= 5

            preds = forecaster.predict(X_test)
            assert preds.shape == (N_test, output_dim)
            assert (preds >= 0).all()

            eval_res = forecaster.evaluate(X_test, y_test)
            assert "t+5min" in eval_res
            assert "t+10min" in eval_res
            assert "t+15min" in eval_res

            # Test save and load
            model_path = os.path.join(temp_dir, "forecasting_lstm.pt")
            forecaster.save(model_path)
            assert os.path.exists(model_path)

            loaded = DeepLearningForecaster.load(model_path)
            loaded_preds = loaded.predict(X_test)
            np.testing.assert_allclose(preds, loaded_preds, rtol=1e-5, atol=1e-5)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestUnifiedTrainAndSave:
    """Tests end-to-end execution of train_and_save."""

    def test_train_and_save_artifacts_created(self):
        temp_dir = tempfile.mkdtemp()
        try:
            results = train_and_save(
                data_dir="data/processed",
                save_dir=temp_dir,
                epochs=3,
                batch_size=16,
            )

            assert "xgboost_model" in results
            assert "lstm_forecaster" in results
            assert "summary" in results

            assert os.path.exists(os.path.join(temp_dir, "forecasting_lstm.pt"))
            assert os.path.exists(os.path.join(temp_dir, "xgboost_baseline_head_0.json"))
            assert os.path.exists(os.path.join(temp_dir, "xgboost_baseline_head_1.json"))
            assert os.path.exists(os.path.join(temp_dir, "training_summary.json"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
