# attacks/live_packet_sniffer.py

import sys
import time
from collections import defaultdict
from scapy.all import IP, TCP, UDP, ICMP, sniff

# Path fix to import dashboard_bridge from parent/sibling directory
sys.path.append("../dashboard")
from dashboard_bridge import bridge_instance

# Configurable Timeouts & Windows (in seconds)
PORT_SCAN_WINDOW = 60  # Time window for tracking port scans
FLOW_TIMEOUT = 30      # Inactivity threshold to consider a flow closed

# Flow Table: key -> {first_seen, last_seen, packet_count, bytes}
active_flows = {}

# Port Tracker: src_ip -> set of (dst_port, timestamp)
port_tracker = defaultdict(set)
packet_counter = 0


def cleanup_stale_entries(current_time):
    """Purge expired flows and stale port tracking data."""
    # 1. Clean up expired port scan records
    for src_ip in list(port_tracker.keys()):
        port_tracker[src_ip] = {
            (port, ts) for port, ts in port_tracker[src_ip]
            if current_time - ts <= PORT_SCAN_WINDOW
        }
        if not port_tracker[src_ip]:
            del port_tracker[src_ip]

    # 2. Clean up inactive 5-tuple flows
    for flow_key in list(active_flows.keys()):
        if current_time - active_flows[flow_key]["last_seen"] > FLOW_TIMEOUT:
            del active_flows[flow_key]


def process_packet(packet):
    global packet_counter
    if not packet.haslayer(IP):
        return

    now = time.time()
    packet_counter += 1
    
    # Run periodic cleanup to release unbounded memory
    cleanup_stale_entries(now)

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    proto = packet[IP].proto
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

    src_port = 0
    dst_port = 0
    tcp_flags = ""

    # Parse L4 Protocols
    if packet.haslayer(TCP):
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
        tcp_flags = str(packet[TCP].flags)
        port_tracker[src_ip].add((dst_port, now))

    elif packet.haslayer(UDP):
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport

    # 1. Construct 5-tuple Flow Key
    flow_key = (src_ip, dst_ip, src_port, dst_port, proto)

    # 2. Aggregate Flow Statistics
    if flow_key not in active_flows:
        active_flows[flow_key] = {
            "first_seen": now,
            "last_seen": now,
            "packet_count": 1,
            "bytes": len(packet),
        }
    else:
        active_flows[flow_key]["last_seen"] = now
        active_flows[flow_key]["packet_count"] += 1
        active_flows[flow_key]["bytes"] += len(packet)

    flow_info = active_flows[flow_key]
    flow_duration = round(flow_info["last_seen"] - flow_info["first_seen"], 3)

    # 3. Time-Bounded Rule-Based Classification
    pred_label = "BENIGN"
    confidence = 0.95

    recent_ports = {port for port, _ in port_tracker[src_ip]}
    
    if len(recent_ports) > 10:
        pred_label = "PortScan"
        confidence = 0.98
    elif tcp_flags == "S":  # Pure TCP SYN Packet
        pred_label = "DoS Attack"  # Update to "DoS Attack?" if DB schema mandates
        confidence = 0.96

    # 4. Construct Telemetry Payload
    flow_data = {
        "timestamp": timestamp_str,
        "src_ip": src_ip,
        "src_port": src_port,
        "dst_ip": dst_ip,
        "dst_port": dst_port,
        "protocol": proto,
        "flow_duration": flow_duration,
        "packet_count": flow_info["packet_count"],
    }

    # Dynamic SHAP explanations using computed flow metrics
    if pred_label == "DoS Attack":
        shap_explanation = {
            "Flow Packets/s": round(flow_info["packet_count"] / max(flow_duration, 0.001), 2),
            "SYN Flag Count": flow_info["packet_count"],
            "Flow Duration": flow_duration,
        }
    elif pred_label == "PortScan":
        shap_explanation = {
            "Distinct Dst Ports": len(recent_ports),
            "Flow Duration": flow_duration,
        }
    else:
        shap_explanation = {
            "Packet Length Mean": round(flow_info["bytes"] / flow_info["packet_count"], 2),
            "Flow Duration": flow_duration,
        }

    # 5. Push Aggregated State to Bridge
    bridge_instance.update_telemetry(
        packet_count=packet_counter,
        active_flows=len(active_flows),  # Accurate 5-tuple flow count
    )
    bridge_instance.push_prediction(
        flow_data=flow_data,
        prediction=pred_label,
        confidence=confidence,
        shap_explanation=shap_explanation,
    )

    print(
        f"[CAPTURED] {src_ip}:{src_port} -> {dst_ip}:{dst_port} | "
        f"Duration: {flow_duration}s | Pkts: {flow_info['packet_count']} | Detected: {pred_label}"
    )


def start_listening():
    print("=" * 60)
    print("     XAI-NIDS Live Packet Sniffer Bridge")
    print("=" * 60)
    print("[+] Listening for live network packets... (Press Ctrl+C to stop)\n")

    sniff(filter="ip", prn=process_packet, store=0)


if __name__ == "__main__":
    try:
        start_listening()
    except PermissionError:
        print("\n[✘] Permission Error: Run this terminal as Administrator!")
    except KeyboardInterrupt:
        print("\n[!] Sniffer stopped.")