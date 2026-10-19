"""
Early-Warning Threshold Engine for Network Attack Forecasting Agent.
(SIH 2026 - SIH26153)

Implements:
1. Dual-mode early-warning detection:
   - Static thresholds (predefined attack flow volume & risk limits)
   - Dynamic statistical thresholds (rolling_mean + 1.5 * rolling_std for YELLOW,
     rolling_mean + 2.5 * rolling_std for RED)
2. Tri-state alert classification:
   - GREEN: Normal operational baseline
   - YELLOW: Elevated risk forecast in upcoming temporal windows
   - RED: Imminent high-volume attack surge detected in upcoming windows
3. Stream evaluation for continuous time-series forecasting & incident response playbooks.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class AlertResult:
    """Detailed alert decision container."""
    status: str  # "GREEN", "YELLOW", "RED"
    predicted_val: float
    baseline_mean: float
    baseline_std: float
    dynamic_yellow: float
    dynamic_red: float
    static_yellow: float
    static_red: float
    trigger_reason: str


class AlertEngine:
    """
    Early-Warning Threshold & Mitigation Engine supporting static, dynamic,
    and hybrid statistical anomaly detection.
    """

    def __init__(
        self,
        static_yellow_limit: float = 75.0,
        static_red_limit: float = 160.0,
        dynamic_yellow_std_multiplier: float = 1.7,
        dynamic_red_std_multiplier: float = 2.5,
        minimum_attack_baseline: float = 75.0,
        mode: str = "dynamic",  # "dynamic", "hybrid", "static"
    ):
        self.static_yellow_limit = static_yellow_limit
        self.static_red_limit = static_red_limit
        self.dynamic_yellow_std_multiplier = dynamic_yellow_std_multiplier
        self.dynamic_red_std_multiplier = dynamic_red_std_multiplier
        self.minimum_attack_baseline = minimum_attack_baseline
        self.mode = mode.lower()

    def check_threshold(
        self,
        predicted_series: Union[float, int, np.ndarray, pd.Series, List[float]],
        historical_baseline: Union[np.ndarray, pd.Series, List[float]],
    ) -> str:
        """
        Evaluate predicted intensity against historical baseline.

        Args:
            predicted_series: Single predicted value or multi-horizon predictions
                              (e.g., maximum forecasted attack count across upcoming windows).
            historical_baseline: Array/Series of past window values used to compute
                                 rolling mean and standard deviation.

        Returns:
            alert_state: "GREEN", "YELLOW", or "RED".
        """
        detail = self.check_threshold_detail(predicted_series, historical_baseline)
        return detail.status

    def check_threshold_detail(
        self,
        predicted_series: Union[float, int, np.ndarray, pd.Series, List[float]],
        historical_baseline: Union[np.ndarray, pd.Series, List[float]],
    ) -> AlertResult:
        """
        Evaluate predicted intensity and return full statistical justification.
        """
        # Extract forecast value (e.g. maximum of upcoming forecast horizon)
        if isinstance(predicted_series, (list, tuple, np.ndarray, pd.Series)):
            pred_arr = np.asarray(predicted_series, dtype=np.float64)
            pred_val = float(np.max(pred_arr)) if len(pred_arr) > 0 else 0.0
        else:
            pred_val = float(predicted_series)

        # Compute baseline parameters
        base_arr = np.asarray(historical_baseline, dtype=np.float64)
        if len(base_arr) > 1:
            mean = float(np.mean(base_arr))
            std = float(np.std(base_arr))
            std = max(std, 1.0)  # avoid division by zero / vanishing std
        elif len(base_arr) == 1:
            mean = float(base_arr[0])
            std = max(float(base_arr[0]) * 0.1, 1.0)
        else:
            mean = 0.0
            std = 1.0

        # Protect against threshold collapse during idle periods
        dynamic_yellow = max(mean + (self.dynamic_yellow_std_multiplier * std), self.minimum_attack_baseline)
        dynamic_red = max(
            mean + (self.dynamic_red_std_multiplier * std),
            dynamic_yellow + max((self.dynamic_red_std_multiplier - self.dynamic_yellow_std_multiplier) * std, 1.0),
        )

        # Evaluate based on active mode
        is_red = False
        is_yellow = False
        reasons = []

        if self.mode in ["dynamic", "hybrid"]:
            if pred_val >= dynamic_red:
                is_red = True
                reasons.append(f"Exceeded dynamic RED threshold ({pred_val:.1f} >= {dynamic_red:.1f})")
            elif pred_val >= dynamic_yellow:
                is_yellow = True
                reasons.append(f"Exceeded dynamic YELLOW threshold ({pred_val:.1f} >= {dynamic_yellow:.1f})")

        if self.mode in ["static", "hybrid"]:
            if pred_val >= self.static_red_limit:
                is_red = True
                reasons.append(f"Exceeded static RED limit ({pred_val:.1f} >= {self.static_red_limit})")
            elif pred_val >= self.static_yellow_limit and not is_red:
                is_yellow = True
                reasons.append(f"Exceeded static YELLOW limit ({pred_val:.1f} >= {self.static_yellow_limit})")

        if is_red:
            status = "RED"
        elif is_yellow:
            status = "YELLOW"
        else:
            status = "GREEN"
            reasons.append(f"Prediction within normal baseline bounds ({pred_val:.1f})")

        return AlertResult(
            status=status,
            predicted_val=pred_val,
            baseline_mean=round(mean, 2),
            baseline_std=round(std, 2),
            dynamic_yellow=round(dynamic_yellow, 2),
            dynamic_red=round(dynamic_red, 2),
            static_yellow=self.static_yellow_limit,
            static_red=self.static_red_limit,
            trigger_reason="; ".join(reasons),
        )

    def evaluate_forecast_stream(
        self,
        predictions: np.ndarray,
        baseline_history: np.ndarray,
        baseline_lookback: int = 12,
    ) -> List[AlertResult]:
        """
        Evaluate an entire time-series stream of predictions using sliding baseline windows.
        """
        alert_results = []
        num_points = len(predictions)

        for i in range(num_points):
            pred = predictions[i]
            # Use rolling historical window up to step i
            start_idx = max(0, i - baseline_lookback)
            hist = baseline_history[start_idx:i] if i > 0 else np.array([baseline_history[0]])
            if len(hist) == 0:
                hist = np.array([0.0])

            res = self.check_threshold_detail(pred, hist)
            alert_results.append(res)

        return alert_results

    # -------------------------------------------------------------------------
    # Backward Compatibility Methods for app.py
    # -------------------------------------------------------------------------
    def determine_threat_level(self, risk_score: float) -> str:
        """Map risk score to threat level description."""
        if risk_score >= 50.0:
            return "CRITICAL"
        elif risk_score >= 25.0:
            return "HIGH"
        elif risk_score >= 10.0:
            return "ELEVATED"
        elif risk_score > 3.0:
            return "GUARDED"
        return "NORMAL"

    def identify_attack_vector(self, row: Union[pd.Series, Dict[str, Any]]) -> str:
        """Infer attack vector from signature metrics."""
        total_flows = max(1, row.get("total_flows", 1))
        syn_ratio = (row.get("syn_flag_count", 0) / total_flows) * 100
        failed_ratio = (row.get("failed_connections", 0) / total_flows) * 100
        bytes_per_flow = row.get("byte_volume", 0) / total_flows

        if syn_ratio > 40.0 and total_flows > 150:
            return "SYN Flood / Volumetric DDoS"
        elif failed_ratio > 50.0 and row.get("distinct_src_ips", 0) <= 10:
            return "Credential Brute-Force (SSH/RDP)"
        elif failed_ratio > 40.0 and row.get("distinct_src_ips", 0) > 10:
            return "Distributed Port Scan / Reconnaissance"
        elif bytes_per_flow > 2500 and row.get("risk_score", 0) > 30.0:
            return "Volumetric UDP/Application DoS"
        elif row.get("risk_score", 0) > 10.0:
            return "Anomalous Intrusion Traffic"
        return "Baseline Enterprise Traffic"

    def get_mitigation_actions(self, threat_level: str, attack_vector: str) -> List[str]:
        """Generate prescriptive mitigation recommendations."""
        actions = []
        if threat_level in ["CRITICAL", "HIGH", "RED"]:
            if "SYN Flood" in attack_vector:
                actions.extend([
                    "Activate TCP SYN Cookie protection at perimeter firewalls.",
                    "Throttle inbound TCP handshake rate exceeding 100 conns/sec.",
                    "Engage cloud scrubbing and upstream volumetric mitigation.",
                ])
            elif "Brute-Force" in attack_vector:
                actions.extend([
                    "Apply dynamic IP blacklisting via Fail2ban / NGFW rules.",
                    "Enforce MFA challenges for all external SSH and RDP endpoints.",
                    "Limit authentication attempts to 3 per minute per source host.",
                ])
            elif "Port Scan" in attack_vector:
                actions.extend([
                    "Drop ICMP and TCP probing scans from suspicious IP blocks.",
                    "Isolate vulnerable honeypots and verify security group bindings.",
                ])
            else:
                actions.extend([
                    "Deploy automated traffic diversion and inspection rules.",
                    "Alert Tier-2 Incident Response Handlers.",
                ])
        elif threat_level in ["ELEVATED", "YELLOW"]:
            actions.append("Elevate logging frequency and inspect connection velocity per subnet.")
        else:
            actions.append("All network parameters nominal. Continuous passive monitoring active.")

        return actions

    def evaluate_window(self, row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess a single traffic window for the Streamlit dashboard."""
        risk = float(row.get("risk_score", 0.0))
        threat = self.determine_threat_level(risk)
        vector = self.identify_attack_vector(row)
        recs = self.get_mitigation_actions(threat, vector)

        # Also compute alert status
        flow_count = float(row.get("attack_count", 0))
        if flow_count >= self.static_red_limit:
            status = "RED"
        elif flow_count >= self.static_yellow_limit:
            status = "YELLOW"
        else:
            status = "GREEN"

        return {
            "threat_level": threat,
            "alert_status": status,
            "attack_vector": vector,
            "recommendations": recs,
        }

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich a windowed traffic DataFrame with alert states and threat levels."""
        res = df.copy()
        threat_levels = []
        alert_statuses = []
        vectors = []

        for _, row in res.iterrows():
            eval_data = self.evaluate_window(row)
            threat_levels.append(eval_data["threat_level"])
            alert_statuses.append(eval_data["alert_status"])
            vectors.append(eval_data["attack_vector"])

        res["threat_level"] = threat_levels
        res["alert_status"] = alert_statuses
        res["attack_vector"] = vectors
        return res


def check_threshold(
    predicted_series: Union[float, int, np.ndarray, pd.Series, List[float]],
    historical_baseline: Union[np.ndarray, pd.Series, List[float]],
    mode: str = "hybrid",
) -> str:
    """
    Convenience functional interface to evaluate early-warning threshold:
    check_threshold(predicted_series, historical_baseline) -> alert_state
    """
    engine = AlertEngine(mode=mode)
    return engine.check_threshold(predicted_series, historical_baseline)
