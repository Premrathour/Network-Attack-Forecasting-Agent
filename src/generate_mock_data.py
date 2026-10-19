"""
Synthetic Network Flow Generator for Network Attack Forecasting Agent (SIH 2026 - SIH26153).

Generates 24 hours of synthetic network flow logs mimicking CICIDS2017/UNSW-NB15
conventions with baseline enterprise traffic and periodic cyber attack surges (DDoS,
Brute Force, Port Scan, DoS).
"""

import argparse
import os
import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


def generate_mock_flows(
    start_time: Optional[datetime] = None,
    duration_hours: int = 24,
    random_seed: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate synthetic network flow logs with realistic normal patterns and
    simulated attack surges.

    Args:
        start_time: Starting datetime for flow logs (defaults to yesterday 00:00:00).
        duration_hours: Number of hours to simulate (default: 24).
        random_seed: Random seed for reproducibility.
        output_path: Optional file path to save the generated CSV.

    Returns:
        pd.DataFrame containing simulated network flows.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    if start_time is None:
        start_time = datetime(2026, 10, 19, 0, 0, 0)

    end_time = start_time + timedelta(hours=duration_hours)

    # Infrastructure IP profiles
    internal_subnets = [f"192.168.1.{i}" for i in range(10, 80)]
    internal_servers = {
        "web": "10.0.0.5",
        "api": "10.0.0.8",
        "db": "10.0.0.12",
        "gateway": "10.0.0.1",
    }
    public_dns = ["8.8.8.8", "1.1.1.1"]
    trusted_partner_ips = [f"198.51.100.{i}" for i in range(10, 25)]

    # Attack source IPs pools
    brute_force_ips = ["203.0.113.45", "203.0.113.78", "203.0.113.92"]
    port_scan_ips = ["198.51.100.220", "198.51.100.221"]
    botnet_ips = [f"185.{random.randint(10, 250)}.{random.randint(1, 254)}.{random.randint(1, 254)}" for _ in range(120)]

    records = []

    # Iterate minute-by-minute across the entire duration
    current_minute = start_time
    total_minutes = duration_hours * 60

    for minute_idx in range(total_minutes):
        minute_time = current_minute + timedelta(minutes=minute_idx)
        hour_float = minute_time.hour + (minute_time.minute / 60.0)

        # -------------------------------------------------------------
        # Determine Attack Scenario active during this minute
        # -------------------------------------------------------------
        # Phase 1: 03:00 - 04:30 -> SSH & RDP Brute Force
        is_brute_force = 3.0 <= hour_float < 4.5

        # Phase 2: 09:30 - 11:30 -> Volumetric SYN Flood DDoS
        is_ddos_syn = 9.5 <= hour_float < 11.5

        # Phase 3: 15:00 - 16:15 -> Port Scanning Reconnaissance
        is_port_scan = 15.0 <= hour_float < 16.25

        # Phase 4: 20:00 - 21:30 -> High Volume Volumetric DoS
        is_dos_volumetric = 20.0 <= hour_float < 21.5

        # -------------------------------------------------------------
        # 1. Baseline Normal Enterprise Traffic
        # -------------------------------------------------------------
        # Diurnal pattern: traffic peaks during business hours (09:00 - 18:00)
        diurnal_factor = 1.0 + 0.6 * np.sin((hour_float - 6) / 24 * 2 * np.pi)
        base_flows_count = max(8, int(np.random.poisson(25) * diurnal_factor))

        for _ in range(base_flows_count):
            flow_sec = random.randint(0, 59)
            flow_ts = minute_time + timedelta(seconds=flow_sec)
            src_ip = random.choice(internal_subnets)
            dest_choice = random.random()

            if dest_choice < 0.60:
                dst_ip = internal_servers["web"]
                dst_port = random.choice([80, 443])
                proto = "TCP"
            elif dest_choice < 0.80:
                dst_ip = random.choice(public_dns)
                dst_port = 53
                proto = "UDP"
            elif dest_choice < 0.92:
                dst_ip = internal_servers["api"]
                dst_port = 8080
                proto = "TCP"
            else:
                dst_ip = internal_servers["db"]
                dst_port = 5432
                proto = "TCP"

            src_port = random.randint(32768, 61000)
            fwd_bytes = int(np.random.lognormal(mean=6.2, sigma=0.8))  # ~500B
            bwd_bytes = int(np.random.lognormal(mean=7.5, sigma=1.0))  # ~1800B
            syn_flag = 1 if (proto == "TCP" and random.random() < 0.05) else 0
            failed_conn = 1 if random.random() < 0.02 else 0

            records.append({
                "timestamp": flow_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": proto,
                "fwd_bytes": fwd_bytes,
                "bwd_bytes": bwd_bytes,
                "syn_flag": syn_flag,
                "failed_connection": failed_conn,
                "label": "BENIGN",
            })

        # -------------------------------------------------------------
        # 2. Attack Surges Injection
        # -------------------------------------------------------------
        # Attack 1: Brute Force Surge
        if is_brute_force:
            bf_intensity = int(np.random.poisson(45))
            for _ in range(bf_intensity):
                flow_sec = random.randint(0, 59)
                flow_ts = minute_time + timedelta(seconds=flow_sec)
                src_ip = random.choice(brute_force_ips)
                dst_ip = internal_servers["web"]
                dst_port = random.choice([22, 3389])
                fwd_bytes = random.randint(120, 340)
                bwd_bytes = random.randint(80, 200)
                failed_conn = 1 if random.random() < 0.90 else 0  # Failed login attempts

                records.append({
                    "timestamp": flow_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": random.randint(40000, 65000),
                    "dst_port": dst_port,
                    "protocol": "TCP",
                    "fwd_bytes": fwd_bytes,
                    "bwd_bytes": bwd_bytes,
                    "syn_flag": 1 if random.random() < 0.40 else 0,
                    "failed_connection": failed_conn,
                    "label": "Brute Force",
                })

        # Attack 2: Volumetric SYN Flood DDoS
        if is_ddos_syn:
            ddos_intensity = int(np.random.poisson(110))
            for _ in range(ddos_intensity):
                flow_sec = random.randint(0, 59)
                flow_ts = minute_time + timedelta(seconds=flow_sec)
                src_ip = random.choice(botnet_ips)
                dst_ip = internal_servers["web"]
                dst_port = random.choice([80, 443])
                fwd_bytes = random.randint(60, 120)
                bwd_bytes = 0  # Unacknowledged half-open TCP connections
                syn_flag = 1
                failed_conn = 1 if random.random() < 0.96 else 0

                records.append({
                    "timestamp": flow_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": random.randint(1024, 65535),
                    "dst_port": dst_port,
                    "protocol": "TCP",
                    "fwd_bytes": fwd_bytes,
                    "bwd_bytes": bwd_bytes,
                    "syn_flag": syn_flag,
                    "failed_connection": failed_conn,
                    "label": "DDoS",
                })

        # Attack 3: Port Scan Reconnaissance
        if is_port_scan:
            scan_intensity = int(np.random.poisson(35))
            target_ports = [21, 22, 23, 25, 80, 110, 139, 443, 445, 1433, 3306, 3389, 8080, 8443]
            for _ in range(scan_intensity):
                flow_sec = random.randint(0, 59)
                flow_ts = minute_time + timedelta(seconds=flow_sec)
                src_ip = random.choice(port_scan_ips)
                dst_ip = random.choice([internal_servers["web"], internal_servers["api"], internal_servers["db"]])
                dst_port = random.choice(target_ports)
                fwd_bytes = random.randint(40, 80)
                bwd_bytes = random.randint(0, 40)
                syn_flag = 1
                failed_conn = 1 if random.random() < 0.92 else 0

                records.append({
                    "timestamp": flow_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": random.randint(50000, 65000),
                    "dst_port": dst_port,
                    "protocol": "TCP",
                    "fwd_bytes": fwd_bytes,
                    "bwd_bytes": bwd_bytes,
                    "syn_flag": syn_flag,
                    "failed_connection": failed_conn,
                    "label": "PortScan",
                })

        # Attack 4: Volumetric DoS / UDP Flood
        if is_dos_volumetric:
            dos_intensity = int(np.random.poisson(90))
            for _ in range(dos_intensity):
                flow_sec = random.randint(0, 59)
                flow_ts = minute_time + timedelta(seconds=flow_sec)
                src_ip = random.choice(botnet_ips)
                dst_ip = internal_servers["web"]
                dst_port = random.choice([80, 443, 53, 5060])
                fwd_bytes = random.randint(1200, 4500)
                bwd_bytes = random.randint(0, 200)
                syn_flag = 0
                failed_conn = 1 if random.random() < 0.88 else 0

                records.append({
                    "timestamp": flow_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": random.randint(1024, 65535),
                    "dst_port": dst_port,
                    "protocol": random.choice(["UDP", "TCP"]),
                    "fwd_bytes": fwd_bytes,
                    "bwd_bytes": bwd_bytes,
                    "syn_flag": syn_flag,
                    "failed_connection": failed_conn,
                    "label": "DoS",
                })

    df = pd.DataFrame(records)
    # Sort chronologically
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[generate_mock_data] Generated {len(df):,} synthetic flow records saved to: {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic network flow logs.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw_flow_logs.csv",
        help="Path where generated CSV should be saved",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Number of hours of traffic to simulate (default: 24)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    generate_mock_flows(
        duration_hours=args.hours,
        random_seed=args.seed,
        output_path=args.output,
    )
