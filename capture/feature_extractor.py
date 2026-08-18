"""
feature_extractor.py
--------------------
Converts a completed Flow object into the 78-column CIC-IDS 2017
compatible feature dictionary.

The live system uses the same feature names as the trained model while also
preserving packet-flow metadata used by the behavioural detectors and
explainable dashboard.
"""

import math
import statistics
import time
from typing import Any, Dict, List

import numpy as np

from capture.constants import (
    ACTIVITY_TIMEOUT,
    BULK_BOUND,
    FEATURE_COLUMNS,
    TCP_ACK, TCP_CWR, TCP_ECE, TCP_FIN,
    TCP_PSH, TCP_RST, TCP_SYN, TCP_URG,
)
from capture.flow import Flow


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def _stats(values: List[float]) -> tuple:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std()), float(arr.max()), float(arr.min())


def _compute_active_idle(fwd_ts: List[float], bwd_ts: List[float], threshold: float = ACTIVITY_TIMEOUT) -> tuple:
    all_ts = sorted(fwd_ts + bwd_ts)
    if len(all_ts) < 2:
        return [], []
    active_times: List[float] = []
    idle_times: List[float] = []
    period_start = all_ts[0]
    prev_ts = all_ts[0]
    for ts in all_ts[1:]:
        gap = ts - prev_ts
        if gap > threshold:
            active_dur = prev_ts - period_start
            if active_dur > 0:
                active_times.append(active_dur)
            idle_times.append(gap)
            period_start = ts
        prev_ts = ts
    last_active_dur = all_ts[-1] - period_start
    if last_active_dur > 0:
        active_times.append(last_active_dur)
    return active_times, idle_times


def _count_flag(flags_list: List[int], mask: int) -> int:
    return sum(1 for f in flags_list if f & mask)


def extract_features(flow: Flow) -> Dict[str, Any]:
    fts = flow.fwd_timestamps
    bts = flow.bwd_timestamps
    fpl = flow.fwd_pkt_lens
    bpl = flow.bwd_pkt_lens
    fhl = flow.fwd_header_lens
    bhl = flow.bwd_header_lens
    ffl = flow.fwd_flags
    bfl = flow.bwd_flags
    all_pl = fpl + bpl

    duration_us = flow.duration_us
    duration_s = flow.duration_s
    n_fwd = len(fpl)
    n_bwd = len(bpl)
    total_fwd_bytes = sum(fpl)
    total_bwd_bytes = sum(bpl)

    fpl_arr = np.array(fpl, dtype=float) if fpl else np.zeros(1)
    bpl_arr = np.array(bpl, dtype=float) if bpl else np.zeros(1)
    all_arr = np.array(all_pl, dtype=float) if all_pl else np.zeros(1)

    fwd_len_mean = float(fpl_arr.mean())
    fwd_len_std = float(fpl_arr.std())
    fwd_len_max = float(fpl_arr.max())
    fwd_len_min = float(fpl_arr.min())
    bwd_len_mean = float(bpl_arr.mean())
    bwd_len_std = float(bpl_arr.std())
    bwd_len_max = float(bpl_arr.max())
    bwd_len_min = float(bpl_arr.min())
    pkt_len_mean = float(all_arr.mean())
    pkt_len_std = float(all_arr.std())
    pkt_len_max = float(all_arr.max())
    pkt_len_min = float(all_arr.min())
    pkt_len_var = float(all_arr.var())

    total_pkts = n_fwd + n_bwd
    total_bytes = total_fwd_bytes + total_bwd_bytes
    flow_bytes_s = _safe_div(total_bytes, duration_s) if duration_s > 0 else 0.0
    flow_pkts_s = _safe_div(total_pkts, duration_s) if duration_s > 0 else 0.0
    fwd_pkts_s = _safe_div(n_fwd, duration_s) if duration_s > 0 else 0.0
    bwd_pkts_s = _safe_div(n_bwd, duration_s) if duration_s > 0 else 0.0

    all_ts_sorted = sorted(fts + bts)
    flow_iat = [(all_ts_sorted[i] - all_ts_sorted[i - 1]) * 1e6 for i in range(1, len(all_ts_sorted))]
    flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = _stats(flow_iat)
    fwd_iat_us = [(fts[i] - fts[i - 1]) * 1e6 for i in range(1, len(fts))]
    fwd_iat_total = sum(fwd_iat_us)
    fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _stats(fwd_iat_us)
    bwd_iat_us = [(bts[i] - bts[i - 1]) * 1e6 for i in range(1, len(bts))]
    bwd_iat_total = sum(bwd_iat_us)
    bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _stats(bwd_iat_us)

    fin_cnt = _count_flag(ffl + bfl, TCP_FIN)
    syn_cnt = _count_flag(ffl + bfl, TCP_SYN)
    rst_cnt = _count_flag(ffl + bfl, TCP_RST)
    psh_cnt = _count_flag(ffl + bfl, TCP_PSH)
    ack_cnt = _count_flag(ffl + bfl, TCP_ACK)
    urg_cnt = _count_flag(ffl + bfl, TCP_URG)
    cwe_cnt = _count_flag(ffl + bfl, TCP_CWR)
    ece_cnt = _count_flag(ffl + bfl, TCP_ECE)
    fwd_psh_flags = _count_flag(ffl, TCP_PSH)
    bwd_psh_flags = _count_flag(bfl, TCP_PSH)
    fwd_urg_flags = _count_flag(ffl, TCP_URG)
    bwd_urg_flags = _count_flag(bfl, TCP_URG)

    fwd_header_total = sum(fhl)
    bwd_header_total = sum(bhl)
    down_up_ratio = _safe_div(n_bwd, n_fwd)
    avg_pkt_size = _safe_div(total_bytes, total_pkts)
    avg_fwd_seg_size = fwd_len_mean
    avg_bwd_seg_size = bwd_len_mean

    BULK_MIN = 4

    def _bulk_stats(pkt_lens, timestamps):
        bulk_bytes = bulk_pkts = bulk_dur = bulk_count = 0
        run_bytes = run_pkts = 0
        run_first_ts = 0.0
        for i, (plen, ts) in enumerate(zip(pkt_lens, timestamps)):
            if plen > BULK_BOUND:
                if run_pkts == 0:
                    run_first_ts = ts
                run_bytes += plen
                run_pkts += 1
            else:
                if run_pkts >= BULK_MIN:
                    bulk_bytes += run_bytes
                    bulk_pkts += run_pkts
                    bulk_dur += (timestamps[i - 1] - run_first_ts) * 1e6
                    bulk_count += 1
                run_bytes = run_pkts = 0
        if run_pkts >= BULK_MIN:
            bulk_bytes += run_bytes
            bulk_pkts += run_pkts
            if timestamps:
                bulk_dur += (timestamps[-1] - run_first_ts) * 1e6
            bulk_count += 1
        return bulk_bytes, bulk_pkts, bulk_dur, bulk_count

    fb, fp, fd, fc = _bulk_stats(fpl, fts)
    bb, bp, bd, bc = _bulk_stats(bpl, bts)
    fwd_avg_bytes_bulk = _safe_div(fb, fc)
    fwd_avg_packets_bulk = _safe_div(fp, fc)
    fwd_avg_bulk_rate = _safe_div(fb, fd / 1e6) if fd > 0 else 0.0
    bwd_avg_bytes_bulk = _safe_div(bb, bc)
    bwd_avg_packets_bulk = _safe_div(bp, bc)
    bwd_avg_bulk_rate = _safe_div(bb, bd / 1e6) if bd > 0 else 0.0

    _, idle_periods = _compute_active_idle(fts, bts, ACTIVITY_TIMEOUT)
    subflow_count = 1 + len(idle_periods)
    subflow_fwd_pkts = _safe_div(n_fwd, subflow_count)
    subflow_fwd_bytes = _safe_div(total_fwd_bytes, subflow_count)
    subflow_bwd_pkts = _safe_div(n_bwd, subflow_count)
    subflow_bwd_bytes = _safe_div(total_bwd_bytes, subflow_count)

    active_periods, idle_dur_list = _compute_active_idle(fts, bts, ACTIVITY_TIMEOUT)
    active_us = [a * 1e6 for a in active_periods]
    idle_us = [g * 1e6 for g in idle_dur_list]
    act_mean, act_std, act_max, act_min = _stats(active_us)
    idl_mean, idl_std, idl_max, idl_min = _stats(idle_us)

    act_data_fwd = flow.act_data_pkt_fwd
    min_seg_fwd = flow.min_seg_size_fwd if flow.min_seg_size_fwd > 0 else 0

    # Keep high-resolution capture time for behavioural rate calculations.
    # The human-readable timestamp remains available separately.
    capture_ts = float(flow.start_time)

    return {
        "src_ip": flow.src_ip,
        "src_port": flow.src_port,
        "dst_ip": flow.dst_ip,
        "dst_port": flow.dst_port,
        "protocol": flow.protocol,
        "flow_id": flow.flow_id,
        "Destination Port": flow.dst_port,
        "Flow Duration": duration_us,
        "Total Fwd Packets": n_fwd,
        "Total Backward Packets": n_bwd,
        "Total Length of Fwd Packets": total_fwd_bytes,
        "Total Length of Bwd Packets": total_bwd_bytes,
        "Fwd Packet Length Max": fwd_len_max,
        "Fwd Packet Length Min": fwd_len_min,
        "Fwd Packet Length Mean": fwd_len_mean,
        "Fwd Packet Length Std": fwd_len_std,
        "Bwd Packet Length Max": bwd_len_max,
        "Bwd Packet Length Min": bwd_len_min,
        "Bwd Packet Length Mean": bwd_len_mean,
        "Bwd Packet Length Std": bwd_len_std,
        "Flow Bytes/s": flow_bytes_s,
        "Flow Packets/s": flow_pkts_s,
        "Flow IAT Mean": flow_iat_mean,
        "Flow IAT Std": flow_iat_std,
        "Flow IAT Max": flow_iat_max,
        "Flow IAT Min": flow_iat_min,
        "Fwd IAT Total": fwd_iat_total,
        "Fwd IAT Mean": fwd_iat_mean,
        "Fwd IAT Std": fwd_iat_std,
        "Fwd IAT Max": fwd_iat_max,
        "Fwd IAT Min": fwd_iat_min,
        "Bwd IAT Total": bwd_iat_total,
        "Bwd IAT Mean": bwd_iat_mean,
        "Bwd IAT Std": bwd_iat_std,
        "Bwd IAT Max": bwd_iat_max,
        "Bwd IAT Min": bwd_iat_min,
        "Fwd PSH Flags": fwd_psh_flags,
        "Bwd PSH Flags": bwd_psh_flags,
        "Fwd URG Flags": fwd_urg_flags,
        "Bwd URG Flags": bwd_urg_flags,
        "Fwd Header Length": fwd_header_total,
        "Bwd Header Length": bwd_header_total,
        "Fwd Packets/s": fwd_pkts_s,
        "Bwd Packets/s": bwd_pkts_s,
        "Min Packet Length": pkt_len_min,
        "Max Packet Length": pkt_len_max,
        "Packet Length Mean": pkt_len_mean,
        "Packet Length Std": pkt_len_std,
        "Packet Length Variance": pkt_len_var,
        "FIN Flag Count": fin_cnt,
        "SYN Flag Count": syn_cnt,
        "RST Flag Count": rst_cnt,
        "PSH Flag Count": psh_cnt,
        "ACK Flag Count": ack_cnt,
        "URG Flag Count": urg_cnt,
        "CWE Flag Count": cwe_cnt,
        "ECE Flag Count": ece_cnt,
        "Down/Up Ratio": down_up_ratio,
        "Average Packet Size": avg_pkt_size,
        "Avg Fwd Segment Size": avg_fwd_seg_size,
        "Avg Bwd Segment Size": avg_bwd_seg_size,
        "Fwd Header Length.1": fwd_header_total,
        "Fwd Avg Bytes/Bulk": fwd_avg_bytes_bulk,
        "Fwd Avg Packets/Bulk": fwd_avg_packets_bulk,
        "Fwd Avg Bulk Rate": fwd_avg_bulk_rate,
        "Bwd Avg Bytes/Bulk": bwd_avg_bytes_bulk,
        "Bwd Avg Packets/Bulk": bwd_avg_packets_bulk,
        "Bwd Avg Bulk Rate": bwd_avg_bulk_rate,
        "Subflow Fwd Packets": subflow_fwd_pkts,
        "Subflow Fwd Bytes": subflow_fwd_bytes,
        "Subflow Bwd Packets": subflow_bwd_pkts,
        "Subflow Bwd Bytes": subflow_bwd_bytes,
        "Init_Win_bytes_forward": flow.fwd_init_win_bytes,
        "Init_Win_bytes_backward": flow.bwd_init_win_bytes,
        "act_data_pkt_fwd": act_data_fwd,
        "min_seg_size_forward": min_seg_fwd,
        "Active Mean": act_mean,
        "Active Std": act_std,
        "Active Max": act_max,
        "Active Min": act_min,
        "Idle Mean": idl_mean,
        "Idle Std": idl_std,
        "Idle Max": idl_max,
        "Idle Min": idl_min,
        "Label": "BENIGN",
        "capture_ts": capture_ts,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(flow.start_time)),
    }
