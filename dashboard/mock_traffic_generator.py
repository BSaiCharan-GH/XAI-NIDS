# mock_traffic_generator.py

import random
import time
from dashboard_bridge import bridge_instance


def generate_mock_traffic():
    print(
        "[MOCK GENERATOR] Started feeding synthetic traffic into Dashboard Bridge..."
    )

    attack_types = ["DoS Attack", "PortScan", "BENIGN", "BENIGN", "BENIGN"]
    sample_ips = ["192.168.1.10", "192.168.1.15", "10.0.0.5", "172.16.0.22"]

    packet_counter = 0

    while True:
        packet_counter += random.randint(10, 50)
        active_flows = random.randint(3, 12)

        # Update telemetry
        bridge_instance.update_telemetry(
            packet_count=packet_counter, active_flows=active_flows
        )

        pred_label = random.choice(attack_types)
        confidence = (
            round(random.uniform(0.85, 0.99), 4)
            if pred_label != "BENIGN"
            else round(random.uniform(0.92, 0.99), 4)
        )
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

        flow_data = {
            "timestamp": timestamp_str,
            "src_ip": random.choice(sample_ips),
            "src_port": random.randint(1024, 65535),
            "dst_ip": "192.168.1.1",
            "dst_port": (
                80
                if pred_label == "DoS Attack"
                else random.choice([22, 80, 443, 8080])
            ),
            "protocol": 6,
            "Flow Duration": random.randint(1000, 500000),
        }

        if pred_label == "DoS Attack":
            shap_explanation = {
                "Flow Packets/s": 2.45,
                "Fwd Packet Length Max": 1.82,
                "Flow Duration": -0.65,
                "Bwd Packet Length Mean": 0.42,
                "Flow IAT Mean": -0.31,
            }
        elif pred_label == "PortScan":
            shap_explanation = {
                "SYN Flag Count": 3.12,
                "Flow IAT Min": -1.45,
                "Fwd Header Length": 0.88,
                "Total Fwd Packets": -0.52,
                "FIN Flag Count": -0.20,
            }
        else:
            shap_explanation = {
                "Packet Length Mean": -0.85,
                "Flow Bytes/s": -0.62,
                "Flow Duration": -0.41,
                "Init_Win_bytes_forward": 0.15,
                "ACK Flag Count": -0.10,
            }

        # Push to bridge across processes
        bridge_instance.push_prediction(
            flow_data=flow_data,
            prediction=pred_label,
            confidence=confidence,
            shap_explanation=shap_explanation,
        )

        print(
            f"[SENT] {flow_data['timestamp']} | {flow_data['src_ip']} -> {flow_data['dst_ip']} | Label: {pred_label}"
        )
        time.sleep(2)


if __name__ == "__main__":
    generate_mock_traffic()