# Network Attack Forecasting Agent (SIH 2026 — Problem SIH26153 / P25)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-28%20passed-brightgreen.svg)](tests/)

An intelligent cyber threat forecasting and regularized time-series aggregation pipeline designed for enterprise Security Operations Centers (SOCs). 

This pipeline ingests heterogeneous network flow logs (CICIDS2017, UNSW-NB15, or synthetic datasets), regularizes them into fixed continuous temporal windows (5-minute bins), computes multi-dimensional threat velocity features, and generates multi-step early-warning forecasts to enable **proactive, pre-emptive defense** before volumetric floods saturate gateway infrastructure.

---

## 🎯 Problem Statement (SIH26153 - P25)

Traditional Intrusion Detection Systems (IDS) and SIEM dashboards trigger **reactively** after traffic spikes have already overwhelmed network interfaces. The objective of **SIH26153 (P25)** is to construct an end-to-end Machine Learning and Deep Learning system that:
1. Ingests raw network flow records and resamples them into regularized 5-minute time-series bins.
2. Extracts temporal threat signals: autoregressive lags ($t-1 \dots t-12$), rolling statistics ($\mu, \sigma$), and cyclical diurnal indicators.
3. Continuously forecasts future malicious attack flow counts at multi-step lookahead horizons ($t+5\text{ min}$, $t+10\text{ min}$, $t+15\text{ min}$).
4. Employs dynamic statistical thresholding with baseline floor protection to prevent false alarm collapse on idle traffic.
5. Provides SecOps teams an **operational early-warning advance lead time** ($\ge 15\text{ mins}$) with an empirical False Positive Alarm Rate under $15\%$.

---

## 📊 Benchmark Summary: Baseline vs. Deep Learning

Both models were evaluated on the chronological holdout test split (15% unseen continuous timeline) under identical operational criteria:

| Metric | Baseline Model (XGBoost) | Deep Learning Model (PyTorch LSTM) | Target / SecOps SLA | Winning Model |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | 115.84 flows | **69.21 flows** | Minimization ($\le 100$) | **PyTorch LSTM (-40.3% error)** |
| **Root Mean Squared Error (RMSE)** | 199.75 flows | **121.10 flows** | Minimization | **PyTorch LSTM (-39.4% error)** |
| **Coefficient of Determination ($R^2$)** | 0.1100 | **0.6896** | Higher ($> 0.60$) | **PyTorch LSTM (+526% fit)** |
| **Average Warning Lead Time** | 0.0 mins | **+35.0 mins** | $\ge 15.0$ mins | **PyTorch LSTM (+35 min advance warning)** |
| **False Positive Alarm Rate (FPAR)** | 0.0% | **7.7%** | $< 15.0\%$ | **PyTorch LSTM (Within SLA)** |
| **Single-Window Inference Latency** | 2.12 ms | **1.33 ms** | $< 50.0$ ms | **PyTorch LSTM (1.6x faster)** |
| **Dynamic Anomaly Tracking** | Conservative / Flat | **High Temporal Fidelity** | Peak Surge Cresting | **PyTorch LSTM** |

---

## 📁 Repository Structure

```
AI CA Project sem 3/
├── data/
│   ├── raw_flow_logs.csv         # 24-hr ingested/synthetic flow records (63,158 flows)
│   ├── windowed_traffic.csv      # Continuous 5-minute regularized time-series windows (288 bins)
│   └── processed/                # Tabular .npz, 3D sequential tensors & predictions
├── saved_models/
│   ├── forecasting_lstm.pt       # Checkpointed 2-layer PyTorch LSTM weights & scaler params
│   ├── xgboost_baseline_head_*.json # Trained XGBoost multi-horizon models (t+1, t+3)
│   ├── xgboost_baseline_scaler.json # XGBoost target scaler parameters
│   └── training_summary.json     # Model training losses and validation checkpoint history
├── reports/
│   ├── benchmark_summary.md      # Comparative benchmarking table & architectural notes
│   ├── project_report.md         # Formal executive submission write-up
│   ├── forecast_vs_actual.png    # Ground truth vs predicted curves with threshold bounds
│   ├── lead_time_distribution.png# Distribution of advance warning lead times
│   ├── confusion_matrix_alerts.png# Early-warning classification confusion matrix
│   └── forecast_evaluation.png   # Full evaluation scorecard overlay
├── src/
│   ├── __init__.py
│   ├── generate_mock_data.py     # 24h synthetic flow simulator with attack campaigns
│   ├── preprocessing.py          # Schema mapping, timestamp parsing & window aggregation
│   ├── features.py               # Cyclical, lag, rolling features & chronological splitting
│   ├── models.py                 # Multi-horizon XGBoost & 2-layer PyTorch LSTM forecaster
│   ├── alert_engine.py           # Dynamic statistical threshold engine with baseline floor
│   └── evaluate.py               # Benchmarking suite, metrics, latency & plot generators
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py     # Schema mapping, timestamp parsing & window aggregation tests
│   ├── test_features.py          # Feature leakage, lag alignment & tensor creation tests
│   ├── test_models.py            # XGBoost & PyTorch LSTM forward pass, fit, save/load tests
│   └── test_alert_evaluate.py    # Static/dynamic thresholds, lead time & FPAR tests
├── app.py                        # Streamlit SOC Command Center Dashboard
├── requirements.txt              # Production dependencies
└── README.md
```

---

## 🚀 Quickstart & Usage

### 1. Environment Setup
```bash
# Clone the repository and install requirements
pip install -r requirements.txt
```

### 2. Data Preprocessing & 5-Min Time-Series Aggregation
```bash
# Ingest raw logs (or auto-generate 24h synthetic flow data if missing)
python src/preprocessing.py --input data/raw_flow_logs.csv --output data/windowed_traffic.csv --window 5min
```

### 3. Feature Engineering & Dataset Splitting
```bash
# Generates tabular arrays and 3D sliding sequence tensors in data/processed/
python src/features.py --input data/windowed_traffic.csv --output-dir data/processed
```

### 4. Train Forecasting Models
```bash
# Trains baseline XGBoost and 2-layer PyTorch LSTM with early stopping and target scaling
python src/models.py --data-dir data/processed --save-dir saved_models --epochs 50 --batch-size 16
```

### 5. Run Evaluation & Benchmark Suite
```bash
# Computes MAE, RMSE, Lead Time, FPAR, Latency, and generates reports/ artifacts
python src/evaluate.py --benchmark
```

### 6. Run Unit Tests (28/28 Passing)
```bash
pytest tests/ -v
```

### 7. Launch Interactive Streamlit SOC Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🛡️ Streamlit SOC Command Center (`app.py`)

The web interface is organized into 3 operational tabs:

1. **🛡️ Real-Time SOC Monitor & Simulation**:
   - Live system status pulse badge (`Normal Baseline`, `Warning (Elevated Threat)`, `Critical Surge Anticipated`).
   - Interactive Plotly visualization showing Ground Truth actuals, multi-step forecasts, shaded surge zones, and dynamic threshold bounds.
   - **"▶️ Simulate Live Traffic Stream"**: Iteratively feeds incoming traffic intervals step-by-step, updating the forecast curve smoothly in-place without widget key collisions.
   - Prescriptive threat mitigation playbook table detailing automated firewall actions (SYN cookies, BGP rate-limiting).

2. **📊 Model Comparison & Benchmarks**:
   - Comprehensive comparative KPI scorecards (MAE, RMSE, Lead Time, FPAR, Latency).
   - Embedded interactive benchmark comparison table.
   - High-resolution submission gallery displaying `forecast_vs_actual.png`, `lead_time_distribution.png`, and `confusion_matrix_alerts.png`.

3. **📑 SecOps Playbook & Executive Report**:
   - Full technical assignment write-up loaded directly from `reports/project_report.md`.

---

## 🔬 Key Engineering Highlights

### 1. Dynamic Threshold Baseline Floor Protection
Dynamic alerting thresholds:
$$\text{Threshold} = \max\left(\mu_{\text{rolling}} + 1.7 \cdot \sigma_{\text{rolling}}, \quad \text{Baseline Floor} = 75.0\right)$$
prevent threshold collapse to zero during quiet traffic hours, dropping the False Positive Alarm Rate from $100\%$ down to **$7.7\%$**.

### 2. +35 Minute Warning Lead Time
The PyTorch 2-Layer LSTM maintains recurrent temporal memory over 12 prior historical intervals ($60\text{ minutes}$), detecting subtle momentum shifts before massive packet floods peak, granting SecOps teams a **+35 minute advance buffer** to deploy traffic scrubbing.

---
*Developed for Smart India Hackathon (SIH 2026) — Problem Statement SIH26153 (P25).*
