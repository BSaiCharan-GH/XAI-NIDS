from live_inference import SYNFloodDetector


detector = SYNFloodDetector(
    window_seconds=5.0,
    minimum_syn_packets=20,
    minimum_syn_flows=10,
    minimum_syn_rate=10.0,
    maximum_ack_ratio=0.25,
    cooldown_seconds=10.0,
)


base_flow = {
    "src_ip": "192.168.0.50",
    "dst_ip": "192.168.0.117",
    "protocol": 6,
    "SYN Flag Count": 2,
    "ACK Flag Count": 0,
    "RST Flag Count": 0,
    "Total Fwd Packets": 1,
    "Total Backward Packets": 0,
}


for i in range(15):

    flow = base_flow.copy()

    flow["dst_port"] = 4000 + i

    result = detector.process_flow(flow)

    if result is not None:
        print()
        print("Detection result:")
        print(result)

        if result.get("detected"):
            print()
            print("========================================")
            print("SYN FLOOD / DoS DETECTED")
            print("========================================")
            break