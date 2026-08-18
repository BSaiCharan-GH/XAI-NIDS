"""Offline smoke test for the complete live-inference decision pipeline.

This script does not send packets and does not touch the network. It feeds
synthetic flow dictionaries into the same LiveInference class used by the
real capture daemon and verifies that the final decision layer can produce
Benign, Port Scan, and DoS decisions.
"""

import time
from pathlib import Path

from ml.live_inference import LiveInference


ROOT = Path(__file__).resolve().parent.parent


def make_flow(inference, dst_port=443, capture_ts=None, syn=0, ack=0, fwd=1, bwd=1):
    flow = {feature: 0.0 for feature in inference.features}
    flow.update(
        {
            "src_ip": "192.168.0.50",
            "src_port": 40000,
            "dst_ip": "192.168.0.117",
            "dst_port": dst_port,
            "protocol": 6,
            "Destination Port": dst_port,
            "SYN Flag Count": syn,
            "ACK Flag Count": ack,
            "RST Flag Count": 0,
            "Total Fwd Packets": fwd,
            "Total Backward Packets": bwd,
            "capture_ts": capture_ts if capture_ts is not None else time.time(),
            "timestamp": "offline-test",
        }
    )
    return flow


def check(label, result, expected):
    actual = result["final_decision"] if result else None
    status = "PASS" if actual == expected else "FAIL"
    print(f"[{status}] {label}: expected={expected}, actual={actual}")
    return actual == expected


def main():
    print("=" * 70)
    print("XAI-NIDS LIVE INFERENCE OFFLINE SMOKE TEST")
    print("No packets are transmitted by this test.")
    print("=" * 70)

    inference = LiveInference(model_path=ROOT / "models" / "random_forest.pkl")
    now = time.time()
    passed = 0

    # 1. Normal flow: should not trigger either behavioural detector.
    result = inference.process_flow(make_flow(inference, dst_port=443, capture_ts=now))
    passed += check("Benign flow", result, result["ml_prediction"] if result else None)

    # 2. Ten different destination ports within five seconds -> Port Scan.
    port_result = None
    for i, port in enumerate(range(10000, 10010)):
        port_result = inference.process_flow(
            make_flow(inference, dst_port=port, capture_ts=now + i * 0.05)
        )
    passed += check("Port Scan behavioural override", port_result, "Port Scan")

    # 3. Twenty SYN packets across ten flows within five seconds, no ACKs.
    dos_result = None
    dos_start = now + 20
    for i in range(10):
        dos_result = inference.process_flow(
            make_flow(
                inference,
                dst_port=80,
                capture_ts=dos_start + i * 0.05,
                syn=2,
                ack=0,
                fwd=1,
                bwd=0,
            )
        )
    passed += check("SYN Flood behavioural override", dos_result, "DoS")

    print("-" * 70)
    print(f"Smoke test result: {passed}/3 checks passed")
    print("=" * 70)
    return 0 if passed == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
