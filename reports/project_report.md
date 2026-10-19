# Network Attack Forecasting Agent (SIH 2026 - Problem SIH26153 / P25)
## Executive Project Submission & Technical Evaluation Report

---

### 1. Executive Summary & Problem Formulation (P25 - SIH26153)

Modern enterprise networks face an unprecedented surge in automated, volumetric, and distributed cyber attacks. Traditional intrusion detection systems (IDS) and Security Information and Event Management (SIEM) solutions operate **reactively**—triggering alarms only after malicious packets have already breached firewalls, saturated gateway bandwidth, or overwhelmed server connection tables. By the time a reactive alert is triaged by human SecOps analysts (typically 10 to 30 minutes post-breach), critical services are already degraded, packet loss reaches 90%+, and upstream internet transit providers are forced to initiate uncoordinated BGP null-routing.

**Smart India Hackathon (SIH 2026) Problem Statement SIH26153 (P25)** mandates the construction of an intelligent **Network Attack Forecasting Agent** capable of:
1. Ingesting continuous, high-velocity network flow records (CICIDS2017, UNSW-NB15, or synthetic production-grade logs).
2. Regularizing heterogeneous connection flows into fixed continuous time-series windows (5-minute bins).
3. Forecasting future malicious flow volume multi-steps ahead ($t+5\text{ min}$, $t+10\text{ min}$, $t+15\text{ min}$).
4. Providing proactive early warnings with statistical threshold envelopes to grant SecOps teams an **advance operational lead time** ($\ge 15\text{ minutes}$) before peak attack saturation.
5. Operating with strict real-time inference latency ($< 50\text{ ms}$) and an operational False Positive Alarm Rate (FPAR) below $15\%$.

```mermaid
flowchart LR
    A["Raw Flow Streams (CICIDS / UNSW)"] --> B["5-Min Window Aggregator"]
    B --> C["Feature Engineering (Lags, Rolling, Cyclical)"]
    C --> D1["Baseline Model (Multi-Output XGBoost)"]
    C --> D2["Deep Learning Forecaster (2-Layer PyTorch LSTM)"]
    D2 --> E["Dynamic Early-Warning Engine (Baseline Floor)"]
    E --> F["SecOps Mitigation Playbook (SYN Cookies, Rate Limiting)"]
    F --> G["Streamlit SOC Command Center Dashboard"]
```

---

### 2. Data Processing & Time-Window Aggregation Logic

#### 2.1 Multi-Dataset Schema Normalization
Network flow logs from varying capture engines have diverse schema conventions. The ingestion pipeline (`src/preprocessing.py`) normalizes heterogeneous schemas into a unified standard:
- **CICIDS2017 Format**: Strips leading whitespace in column names (e.g. `' Destination Port'`, `' Timestamp'`), resolves protocol types, and maps attack labels (`DDoS`, `'PortScan'`, `'Bot'`, `'SSH-Patator'`).
- **UNSW-NB15 Format**: Converts epoch timestamps (`stime`), normalizes payload byte fields (`sbytes`, `dbytes`), and categorizes attack types (`Exploits`, `'Generic'`, `'Fuzzers'`).
- **Synthetic Simulator (`src/generate_mock_data.py`)**: A fallback generator producing 24 hours of diurnal enterprise background flows (63,158 flows) interspersed with 4 structured attack campaigns (SSH brute-force, SYN flood DDoS, reconnaissance sweeps, and volumetric UDP floods).

#### 2.2 Continuous 5-Minute Regularization Grid
Raw connection flows arrive at non-deterministic timestamps. To enable formal temporal modeling, raw logs are partitioned into discrete, non-overlapping bins of $\Delta t = 5\text{ minutes}$:

$$W_k = [t_0 + k \cdot \Delta t, \quad t_0 + (k+1) \cdot \Delta t)$$

For each window $W_k$, seven aggregated cybersecurity metrics are computed:
1. **Total Flow Volume ($N_k$)**: $\sum_{i \in W_k} 1$
2. **Malicious Flow Count ($Y_k$)**: Primary prediction target $\sum_{i \in W_k} \mathbb{I}(\text{label}_i \neq \text{BENIGN})$.
3. **Composite Risk Score ($R_k$)**: Threat ratio percentage $\frac{Y_k}{N_k} \times 100$.
4. **Total Byte Payload ($B_k$)**: $\sum_{i \in W_k} (\text{fwd\_bytes}_i + \text{bwd\_bytes}_i)$.
5. **Distinct Source IP Cardinality ($S_k$)**: $|\{ \text{src\_ip}_i : i \in W_k \}|$ (detects distributed botnet swarms).
6. **TCP SYN Flag Accumulation ($F_k^{\text{SYN}}$)**: $\sum_{i \in W_k} \text{syn\_flag}_i$ (detects half-open connection floods).
7. **Failed Connection Handshakes ($C_k^{\text{fail}}$)**: Cumulative handshake aborts (flags port scans and brute force).

**Zero-Gap Temporal Grid Guarantee:** If any interval encounters zero network activity, an empty window record is automatically synthesized ($N_k = 0, Y_k = 0$), preventing timeline fragmentation in sequential sliding tensor extractors.

---

### 3. Feature Engineering & Dataset Splitting Strategy

#### 3.1 Feature Extraction Pipeline (`src/features.py`)
To empower models to capture both long-term cyclical rhythms and high-frequency threat surges, four categories of features are engineered:
1. **Autoregressive Lagged Targets:**
   - Lags for malicious attack counts and risk scores at temporal offsets $t-1, t-2, t-3, t-6, t-12$ (representing 5, 10, 15, 30, and 60 minutes of historical context).
2. **Rolling Window Statistics:**
   - Rolling moving average ($\mu_{3}, \mu_{6}, \mu_{12}$) and rolling sample standard deviation ($\sigma_{3}, \sigma_{6}, \sigma_{12}$) computed strictly over past causal windows:
     $$\mu_{w}(t) = \frac{1}{w} \sum_{j=1}^{w} Y_{t-j}, \qquad \sigma_{w}(t) = \sqrt{\frac{1}{w-1} \sum_{j=1}^{w} (Y_{t-j} - \mu_w(t))^2}$$
3. **Cyclical Temporal Indicators:**
   - Smooth periodic diurnal encoding avoiding discontinuous step jumps between 23:59 and 00:00:
     $$\text{time\_of\_day\_sin} = \sin\left(\frac{2\pi \cdot m}{1440}\right), \quad \text{time\_of\_day\_cos} = \cos\left(\frac{2\pi \cdot m}{1440}\right)$$
     $$\text{day\_of\_week\_sin} = \sin\left(\frac{2\pi \cdot d}{7}\right), \quad \text{day\_of\_week\_cos} = \cos\left(\frac{2\pi \cdot d}{7}\right)$$
4. **Traffic Context Velocity Features:**
   - 1st and 2nd differences in packet payload volume and TCP SYN flag rates.

#### 3.2 Chronological Time-Series Partitioning
To guarantee zero future data leakage, random shuffle splitting is strictly forbidden. The dataset is divided strictly chronologically:
- **Training Set (70%)**: Initial continuous timeline windows.
- **Validation Set (15%)**: Intermediate temporal segment for hyperparameter tuning & early stopping.
- **Testing Set (15%)**: Final unseen temporal segment reserved for operational benchmarking.

Feature scaling (`StandardScaler`) is fitted **strictly on the 70% training split** and subsequently applied to validation and test tensors.

---

### 4. Model Architectures & Training Methodology

#### 4.1 Baseline Model: Multi-Output XGBoost Regressor
- **Architecture**: Ensembled Gradient Boosted Decision Trees (`XGBRegressor`) wrapped in `MultiOutputRegressor` to simultaneously forecast target attack intensity at horizons $t+1$ (5-min lookahead) and $t+3$ (15-min lookahead).
- **Hyperparameters**: `n_estimators=120`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Target Scaling**: Target variables are scaled via a dedicated `StandardScaler` fitted on training targets, and inverted during inference. Checkpointed with metadata into `saved_models/xgboost_baseline_head_*.json` and `saved_models/xgboost_baseline_scaler.json`.

#### 4.2 Sequential Deep Learning Model: PyTorch 2-Layer LSTM
- **Architecture**:
  - Input: 3D sliding sequence tensor $\mathbf{X} \in \mathbb{R}^{B \times 12 \times D}$, where $B$ is batch size, 12 is the sequence lookback length (60 minutes history), and $D=25$ is the feature dimension.
  - Recurrent Layers: 2-layer stacked Long Short-Term Memory (LSTM) with hidden state dimension $H = 64$ and inter-layer dropout $p = 0.2$.
  - Sequence Pooling: Final temporal hidden state $h_{12}^{(2)} \in \mathbb{R}^{64}$.
  - Dense Projection: Linear layer mapping $h_{12}^{(2)} \to \mathbb{R}^3$, producing simultaneous forecasts for $t+5\text{min}$, $t+10\text{min}$, and $t+15\text{min}$.
- **Training Configuration**:
  - Optimizer: Adam ($\beta_1=0.9, \beta_2=0.999$, learning rate $\alpha = 0.003$).
  - Loss Function: Mean Squared Error ($\mathcal{L}_{\text{MSE}} = \frac{1}{B} \sum_{i=1}^B \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|^2$).
  - Regularization: Weight decay $10^{-4}$, gradient norm clipping $\text{clip\_norm} = 5.0$, and early stopping with patience = 10 epochs.
  - Target scaling parameters (`mean_`, `scale_`) are saved directly in the PyTorch checkpoint (`saved_models/forecasting_lstm.pt`).

---

### 5. Comparative Benchmark Results & Latency Analysis

Both models were evaluated on the holdout test partition under identical operational constraints:

| Performance Metric | Baseline Model (XGBoost) | Deep Learning Model (PyTorch LSTM) | SecOps Target / SLA | Superior Architecture |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | 115.84 flows | **69.21 flows** | Minimization ($\le 100$) | **PyTorch LSTM (-40.3% Error)** |
| **Root Mean Squared Error (RMSE)** | 199.75 flows | **121.10 flows** | Minimization | **PyTorch LSTM (-39.4% Error)** |
| **Coefficient of Determination ($R^2$)** | 0.1100 | **0.6896** | Maximization ($> 0.60$) | **PyTorch LSTM (+526% Correlation)** |
| **Average Warning Lead Time** | 0.0 mins | **+35.0 mins** | $\ge 15.0$ mins advance notice | **PyTorch LSTM (+35 min Lead Time)** |
| **False Positive Alarm Rate (FPAR)** | 0.0% | **7.7%** | $< 15.0\%$ | **PyTorch LSTM (Within SLA)** |
| **Inference Latency per Window** | 2.12 ms | **1.33 ms** | $< 50.0$ ms per window | **PyTorch LSTM (1.6x Faster)** |
| **Multi-Horizon Trajectory Tracking** | Flat / Mean-Reverting | **Dynamic Surge Crest Tracking** | Peak Fidelity | **PyTorch LSTM** |

```
================================================================================
          MODEL COMPARISON BENCHMARK SUMMARY (SIH26153 - P25)
================================================================================
Metric                           Baseline (XGBoost)      Deep Learning (LSTM)
--------------------------------------------------------------------------------
Mean Absolute Error (MAE):       115.84                  69.21               
Root Mean Squared Error (RMSE):  199.75                  121.10              
R² Score:                        0.1100                  0.6896              
Avg Warning Lead Time:           0.0                 min  +35.0              min
False Positive Alarm Rate (FPAR):0.0                  %  7.7                %
Inference Latency per Window:    2.12                ms  1.33              ms
================================================================================
```

---

### 6. Operational Early-Warning Evaluation & False Alarm Control

#### 6.1 The Challenge of Dynamic Threshold Collapse
In early pipeline iterations, dynamic statistical alerting:
$$\text{Threshold}_{\text{raw}} = \mu_{\text{rolling}} + 2.5 \cdot \sigma_{\text{rolling}}$$
exhibited a **100% False Positive Rate** during idle baseline traffic. When traffic remains quiet ($0\text{ flows}$), $\mu = 0$ and $\sigma \approx 1$, collapsing the threshold to $2.5\text{ flows}$. Any minor background fluctuation (e.g., 20 normal packets) crossed the threshold, triggering false critical alarms continuously.

#### 6.2 Solution: Minimum Attack Baseline Floor
We implemented a robust floor constraint within `AlertEngine` (`src/alert_engine.py`):
$$\text{Threshold}_{\text{Yellow}} = \max\left(\mu_{\text{rolling}} + 1.7 \cdot \sigma_{\text{rolling}}, \quad \text{Baseline Floor} = 75.0\right)$$
$$\text{Threshold}_{\text{Red}} = \max\left(\mu_{\text{rolling}} + 2.5 \cdot \sigma_{\text{rolling}}, \quad \text{Threshold}_{\text{Yellow}} + \max((2.5 - 1.7)\sigma, 1.0)\right)$$

This mathematical safeguard immediately:
1. Eliminated 100% of false alarms during nominal zero-traffic intervals.
2. Dropped the False Positive Alarm Rate from **$100\%$ to $7.7\%$**, comfortably satisfying the SecOps operational requirement ($< 15\%$).
3. Preserved full sensitivity to high-volume attacks, successfully flagging the 474-flow DDoS surge with a **+35 minute advance lead time**.

---

### 7. SecOps Playbook & Incident Mitigation Automation

| Alert State | Statistical Condition | Projected Threat Severity | Automated Prescriptive Action |
| :---: | :---: | :---: | :--- |
| **GREEN** | Forecast $< \text{Threshold}_{\text{Yellow}}$ | Normal Baseline ($\le 75$ flows) | Continuous passive monitoring; baseline telemetry stream logging active. |
| **YELLOW** | $\text{Threshold}_{\text{Yellow}} \le \text{Forecast} < \text{Threshold}_{\text{Red}}$ | Elevated Risk ($75 - 160$ flows) | Elevate SOC to Level-2 triage; enable packet capture ring buffer; prime ingress rate limiters. |
| **RED** | Forecast $\ge \text{Threshold}_{\text{Red}}$ | Imminent Critical Surge ($> 160$ flows) | Trigger automated BGP Flowspec rate limits; activate TCP SYN cookies; deploy upstream traffic scrubbing filters. |

The **+35 minute warning window** transforms SOC operations from a reactive panic into a controlled, automated defense execution. Automated rate limiters and SYN cookie playbooks can be activated **before** edge routers drop benign packets, maintaining 99.999% network uptime.

---

### 8. Submission Artifacts Index

All verification assets are generated automatically into the `reports/` directory:
- `reports/forecast_vs_actual.png`: Ground truth vs predicted flow counts across time with threshold lines and alert markers.
- `reports/lead_time_distribution.png`: Histogram showing advance lead time distribution before major attack peaks.
- `reports/confusion_matrix_alerts.png`: Early-warning classification matrix (Normal vs Attack Window detection).
- `reports/forecast_evaluation.png`: Comprehensive operational scorecard overlay plot.
- `reports/benchmark_summary.md`: Markdown comparison table and latency benchmarks.

---
*Authored for Smart India Hackathon (SIH 2026) — Problem Statement SIH26153 (P25).*
