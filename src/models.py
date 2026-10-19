"""
Machine Learning & Deep Learning Forecasting Models for Network Attack Forecasting Agent.
(SIH 2026 - SIH26153)

Implements:
1. Baseline Model:
   - Multi-horizon XGBoost Regressor trained on tabular lag features to predict
     attack_count at t+1 and t+3 steps ahead.
2. Sequential Deep Learning Model (PyTorch):
   - 2-layer LSTM/GRU Recurrent Neural Network with Dropout(0.2) and Linear projection
   - Multi-step forecast predicting the next 3 intervals (t+5min, t+10min, t+15min)
   - Training loop with Adam optimizer, MSE loss, and Early Stopping
   - Weight checkpointing to saved_models/forecasting_lstm.pt
3. Unified train_and_save() entrypoint orchestrating data loading, dual-model training,
   metric logging, and artifact persistence.
"""

import argparse
import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("models")


# =============================================================================
# 1. Baseline Model: Multi-Horizon XGBoost Regressor
# =============================================================================
class XGBoostMultiHorizonForecaster:
    """
    Baseline Regressor using XGBoost with tabular lag features to forecast
    attack counts at t+1 and t+3 intervals ahead.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        base_estimator = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        self.model = MultiOutputRegressor(base_estimator)
        self.target_names: List[str] = ["t+1", "t+3"]
        self.target_scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostMultiHorizonForecaster":
        """
        Train multi-target XGBoost on tabular features with target scaling.
        y_train is expected to contain targets for [t+1, t+3].
        """
        # If y_train has 3 columns (e.g. [t+1, t+2, t+3]), slice columns 0 (t+1) and 2 (t+3)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        if y_train.shape[1] == 3:
            y_train_fit = y_train[:, [0, 2]]
        else:
            y_train_fit = y_train

        # Fit target scaler strictly on training split
        y_train_fit_scaled = self.target_scaler.fit_transform(y_train_fit)

        logger.info(f"Fitting XGBoost Baseline on {X_train.shape[0]} samples with {X_train.shape[1]} tabular features...")
        self.model.fit(X_train, y_train_fit_scaled)
        logger.info("XGBoost Baseline fitting complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict attack count for t+1 and t+3 with inverse scaling."""
        preds_scaled = self.model.predict(X)
        if hasattr(self.target_scaler, "mean_") and self.target_scaler.mean_ is not None:
            preds = self.target_scaler.inverse_transform(preds_scaled)
        else:
            preds = preds_scaled
        return np.clip(preds, a_min=0.0, a_max=None)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute MAE, RMSE, and R2 score for each forecast horizon."""
        if y_true.shape[1] == 3:
            y_eval = y_true[:, [0, 2]]
        else:
            y_eval = y_true

        preds = self.predict(X)
        results = {}
        for idx, horizon in enumerate(self.target_names):
            y_t = y_eval[:, idx]
            y_p = preds[:, idx]

            mae = float(mean_absolute_error(y_t, y_p))
            rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
            r2 = float(r2_score(y_t, y_p))
            results[horizon] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

        return results

    def save(self, filepath: str) -> None:
        """Save model estimators and target scaler to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Save each estimator in multi-output model
        for idx, est in enumerate(self.model.estimators_):
            sub_path = f"{filepath}_head_{idx}.json"
            est.save_model(sub_path)
        if hasattr(self.target_scaler, "mean_") and self.target_scaler.mean_ is not None:
            scaler_data = {
                "mean": self.target_scaler.mean_.tolist(),
                "scale": self.target_scaler.scale_.tolist(),
            }
            with open(f"{filepath}_scaler.json", "w") as f:
                json.dump(scaler_data, f, indent=4)
        logger.info(f"Saved XGBoost Baseline model heads to: {filepath}_head_*.json")

    @classmethod
    def load(cls, filepath: str) -> "XGBoostMultiHorizonForecaster":
        """Load trained XGBoost heads and restore target scaler from disk."""
        forecaster = cls()
        estimators = []
        idx = 0
        while True:
            sub_path = f"{filepath}_head_{idx}.json"
            if not os.path.exists(sub_path):
                break
            booster = XGBRegressor()
            booster.load_model(sub_path)
            estimators.append(booster)
            idx += 1
        if estimators:
            forecaster.model.estimators_ = estimators
        scaler_path = f"{filepath}_scaler.json"
        if os.path.exists(scaler_path):
            with open(scaler_path, "r") as f:
                scaler_data = json.load(f)
            forecaster.target_scaler.mean_ = np.array(scaler_data["mean"])
            forecaster.target_scaler.scale_ = np.array(scaler_data["scale"])
            forecaster.target_scaler.var_ = forecaster.target_scaler.scale_ ** 2
            forecaster.target_scaler.n_samples_seen_ = 100
        return forecaster


# =============================================================================
# 2. Sequential Deep Learning Model: PyTorch Multi-Step LSTM
# =============================================================================
class AttackLSTMNetwork(nn.Module):
    """
    2-layer LSTM Recurrent Neural Network for multi-step network attack forecasting:
    - 2-layer LSTM with inter-layer dropout
    - Dropout(0.2) on final sequence hidden state
    - Linear projection to 3 future time intervals (t+5min, t+10min, t+15min)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 3,
        cell_type: str = "lstm",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.cell_type = cell_type.lower()

        if self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim).
        Returns:
            preds: Output tensor of shape (batch_size, output_dim=3).
        """
        # out shape: (batch_size, sequence_length, hidden_dim)
        out, _ = self.rnn(x)
        # Extract representation from the final sequence timestep
        final_state = out[:, -1, :]
        dropped = self.dropout(final_state)
        preds = self.fc(dropped)
        return preds


class DeepLearningForecaster:
    """
    High-level manager for training, evaluating, and checkpointing the PyTorch
    multi-step sequential recurrent model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 3,
        cell_type: str = "lstm",
        learning_rate: float = 0.002,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = AttackLSTMNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=output_dim,
            cell_type=cell_type,
        ).to(self.device)

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.best_model_state: Optional[Dict[str, Any]] = None
        self.training_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.target_scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 60,
        batch_size: int = 16,
        patience: int = 15,
    ) -> "DeepLearningForecaster":
        """
        Train the recurrent network with target scaling and early stopping on validation loss.
        """
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)

        # Ensure targets have 3 output horizons (t+1, t+2, t+3)
        if y_train.shape[1] == 2:
            y_train = np.column_stack([y_train[:, 0], y_train[:, 0], y_train[:, 1]])
        if y_val.shape[1] == 2:
            y_val = np.column_stack([y_val[:, 0], y_val[:, 0], y_val[:, 1]])

        # Fit target scaler strictly on training split
        y_train_scaled = self.target_scaler.fit_transform(y_train)
        y_val_scaled = self.target_scaler.transform(y_val)

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train_scaled, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val_scaled, dtype=torch.float32),
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(
            f"Starting PyTorch {self.model.cell_type.upper()} training for up to {epochs} epochs "
            f"(Patience: {patience}, Device: {self.device})..."
        )

        for epoch in range(1, epochs + 1):
            # Training phase
            self.model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            epoch_train_loss = float(np.mean(train_losses))

            # Validation phase
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    preds = self.model(batch_x)
                    loss = self.criterion(preds, batch_y)
                    val_losses.append(loss.item())

            epoch_val_loss = float(np.mean(val_losses))
            self.training_history["train_loss"].append(epoch_train_loss)
            self.training_history["val_loss"].append(epoch_val_loss)

            # Checkpoint best weights & early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        f"Epoch {epoch:02d}/{epochs:02d} -> Train MSE: {epoch_train_loss:.4f} | "
                        f"Val MSE: {epoch_val_loss:.4f} [Best model saved]"
                    )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        f"Early stopping triggered at Epoch {epoch}. Best Val MSE: {best_val_loss:.4f}"
                    )
                    break

        # Restore best weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model weights from early stopping checkpoint.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on sequential tensor inputs with inverse target scaling."""
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds_scaled = self.model(tensor_x).cpu().numpy()
        if hasattr(self.target_scaler, "mean_") and self.target_scaler.mean_ is not None:
            preds = self.target_scaler.inverse_transform(preds_scaled)
        else:
            preds = preds_scaled
        return np.clip(preds, a_min=0.0, a_max=None)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute multi-step regression metrics (MAE, RMSE, R2) for next 3 intervals."""
        if y_true.shape[1] == 2:
            y_true = np.column_stack([y_true[:, 0], y_true[:, 0], y_true[:, 1]])

        preds = self.predict(X)
        horizon_labels = ["t+5min", "t+10min", "t+15min"]
        results = {}

        for idx, horizon in enumerate(horizon_labels):
            y_t = y_true[:, idx]
            y_p = preds[:, idx]

            mae = float(mean_absolute_error(y_t, y_p))
            rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
            r2 = float(r2_score(y_t, y_p))
            results[horizon] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

        return results

    def save(self, filepath: str) -> None:
        """Serialize trained model weights and target scaler to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "output_dim": self.model.output_dim,
                "cell_type": self.model.cell_type,
                "training_history": self.training_history,
                "target_scaler_mean": self.target_scaler.mean_.tolist() if hasattr(self.target_scaler, "mean_") and self.target_scaler.mean_ is not None else None,
                "target_scaler_scale": self.target_scaler.scale_.tolist() if hasattr(self.target_scaler, "scale_") and self.target_scaler.scale_ is not None else None,
            },
            filepath,
        )
        logger.info(f"Saved PyTorch {self.model.cell_type.upper()} model weights to: {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "DeepLearningForecaster":
        """Load checkpointed model and restore target scaler from disk."""
        checkpoint = torch.load(filepath, map_location="cpu")
        forecaster = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            output_dim=checkpoint["output_dim"],
            cell_type=checkpoint["cell_type"],
            device=device,
        )
        forecaster.model.load_state_dict(checkpoint["model_state_dict"])
        forecaster.training_history = checkpoint.get("training_history", {})
        if checkpoint.get("target_scaler_mean") is not None:
            forecaster.target_scaler.mean_ = np.array(checkpoint["target_scaler_mean"])
            forecaster.target_scaler.scale_ = np.array(checkpoint["target_scaler_scale"])
            forecaster.target_scaler.var_ = forecaster.target_scaler.scale_ ** 2
            forecaster.target_scaler.n_samples_seen_ = 100
        return forecaster


# =============================================================================
# 3. Unified train_and_save() Entrypoint
# =============================================================================
def train_and_save(
    data_dir: str = "data/processed",
    save_dir: str = "saved_models",
    epochs: int = 50,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Unified training and evaluation pipeline:
    1. Ingests tabular and 3D sequence datasets from data/processed/.
    2. Trains the baseline multi-horizon XGBoost Regressor.
    3. Trains the 2-layer PyTorch LSTM multi-step forecaster.
    4. Evaluates test set performance for both models.
    5. Saves weights and training summary artifacts to saved_models/.
    """
    os.makedirs(save_dir, exist_ok=True)

    tab_path = os.path.join(data_dir, "tabular_features.npz")
    seq_path = os.path.join(data_dir, "sequence_tensors.npz")

    if not os.path.exists(tab_path) or not os.path.exists(seq_path):
        raise FileNotFoundError(
            f"Processed data files not found in '{data_dir}'. Please run src/features.py first."
        )

    logger.info(f"Loading tabular arrays from {tab_path}...")
    tab_data = np.load(tab_path)
    X_train_tab, y_train_tab = tab_data["X_train"], tab_data["y_train"]
    X_val_tab, y_val_tab = tab_data["X_val"], tab_data["y_val"]
    X_test_tab, y_test_tab = tab_data["X_test"], tab_data["y_test"]

    logger.info(f"Loading 3D sequence tensors from {seq_path}...")
    seq_data = np.load(seq_path)
    X_train_seq, y_train_seq = seq_data["X_train"], seq_data["y_train"]
    X_val_seq, y_val_seq = seq_data["X_val"], seq_data["y_val"]
    X_test_seq, y_test_seq = seq_data["X_test"], seq_data["y_test"]

    # -------------------------------------------------------------
    # Step A: Train Baseline XGBoost
    # -------------------------------------------------------------
    logger.info("=== STEP 1/2: Training Baseline XGBoost Model ===")
    xgb_model = XGBoostMultiHorizonForecaster(n_estimators=120, max_depth=5, learning_rate=0.06)
    xgb_model.fit(X_train_tab, y_train_tab, X_val_tab, y_val_tab)
    xgb_test_metrics = xgb_model.evaluate(X_test_tab, y_test_tab)
    logger.info(f"XGBoost Test Metrics: {xgb_test_metrics}")

    xgb_save_prefix = os.path.join(save_dir, "xgboost_baseline")
    xgb_model.save(xgb_save_prefix)

    # -------------------------------------------------------------
    # Step B: Train Deep Learning LSTM
    # -------------------------------------------------------------
    logger.info("=== STEP 2/2: Training Sequential PyTorch LSTM Model ===")
    feature_dim = X_train_seq.shape[2]
    lstm_forecaster = DeepLearningForecaster(
        input_dim=feature_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        output_dim=3,
        cell_type="lstm",
        learning_rate=0.003,
    )
    lstm_forecaster.fit(
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq,
        epochs=epochs,
        batch_size=batch_size,
        patience=10,
    )

    lstm_test_metrics = lstm_forecaster.evaluate(X_test_seq, y_test_seq)
    logger.info(f"PyTorch LSTM Test Metrics: {lstm_test_metrics}")

    lstm_save_path = os.path.join(save_dir, "forecasting_lstm.pt")
    lstm_forecaster.save(lstm_save_path)

    # -------------------------------------------------------------
    # Step C: Save Training Summary & Performance Report
    # -------------------------------------------------------------
    summary = {
        "dataset": {
            "tabular_shapes": {
                "train": list(X_train_tab.shape),
                "val": list(X_val_tab.shape),
                "test": list(X_test_tab.shape),
            },
            "sequential_shapes": {
                "train": list(X_train_seq.shape),
                "val": list(X_val_seq.shape),
                "test": list(X_test_seq.shape),
            },
        },
        "baseline_xgboost": {
            "model_type": "MultiOutput(XGBRegressor)",
            "test_metrics": xgb_test_metrics,
        },
        "deep_learning_lstm": {
            "model_architecture": "2-layer LSTM -> Dropout(0.2) -> Linear(3)",
            "target_horizons": ["t+5min", "t+10min", "t+15min"],
            "test_metrics": lstm_test_metrics,
            "best_train_loss": min(lstm_forecaster.training_history["train_loss"])
            if lstm_forecaster.training_history["train_loss"]
            else None,
            "best_val_loss": min(lstm_forecaster.training_history["val_loss"])
            if lstm_forecaster.training_history["val_loss"]
            else None,
        },
    }

    summary_path = os.path.join(save_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved Training Summary to: {summary_path}")

    return {
        "xgboost_model": xgb_model,
        "lstm_forecaster": lstm_forecaster,
        "summary": summary,
    }


# =============================================================================
# Backward Compatibility Helpers for app.py
# =============================================================================
def build_time_series_features(df: pd.DataFrame, target_col: str = "attack_count", **kwargs):
    """Retained for dashboard backward compatibility."""
    from src.features import TimeSeriesFeatureEngineer
    engineer = TimeSeriesFeatureEngineer(primary_target=target_col)
    artifacts = engineer.process_and_split(df)
    return artifacts["train_df"][artifacts["feature_columns"]], artifacts["train_df"][f"{target_col}_target_t1"]


class AttackTimeSeriesForecaster:
    """Wrapper exposing standard fit/predict for interactive dashboard tabs."""

    def __init__(self, model_type: str = "xgboost", random_state: int = 42):
        self.model_type = model_type
        if model_type == "xgboost":
            self.model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=random_state)
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X[self.feature_names])
        return np.clip(preds, a_min=0, a_max=None)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        preds = self.predict(X)
        mae = float(mean_absolute_error(y, preds))
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        r2 = float(r2_score(y, preds))
        return {"mae": mae, "rmse": rmse, "r2": r2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline XGBoost and PyTorch LSTM models.")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory with processed features")
    parser.add_argument("--save-dir", type=str, default="saved_models", help="Directory to save trained models")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs for LSTM")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for LSTM")
    args = parser.parse_args()

    train_and_save(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
