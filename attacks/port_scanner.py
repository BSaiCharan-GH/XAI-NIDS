# attacks/port_scanner.py

import argparse
import random
import time
from scapy.all import IP, TCP, send


def run_syn_port_scan(
    target_ip: str, start_port: int, end_port: int, delay: float
):
    """Executes a TCP SYN stealth port scan across a range of destination ports."""
    ports = list(range(start_port, end_port + 1))
    random.shuffle(ports)  # Randomize port order to mimic evasive scanning

    total_ports = len(ports)
    print(
        f"\n[+] Launching TCP SYN Port Scan on {target_ip} (Ports {start_port}-{end_port})"
    )
    print(f"[+] Total Ports to Probe: {total_ports} | Delay: {delay}s\n")

    scanned_count = 0
    for port in ports:
        src_port = random.randint(1024, 65535)
        # Craft TCP SYN probe packet
        syn_pkt = IP(dst=target_ip) / TCP(
            sport=src_port, dport=port, flags="S"
        )

        send(syn_pkt, verbose=0)

        scanned_count += 1
        if scanned_count % 25 == 0 or scanned_count == total_ports:
            print(
                f"  [PORT SCAN] Probed {scanned_count}/{total_ports} ports..."
            )

        if delay > 0:
            time.sleep(delay)

    print(
        f"\n[✔] Port Scan completed. Probed {total_ports} target ports on {target_ip}."
    )


def run_quick_scan(target_ip: str, delay: float):
    """Scans top 19 common service ports quickly."""
    common_ports = [
        21,
        22,
        23,
        25,
        53,
        80,
        110,
        135,
        139,
        143,
        443,
        445,
        1433,
        1521,
        3306,
        3389,
        5432,
        8080,
        8443,
    ]
    print(
        f"\n[+] Launching Quick Scan on {target_ip} ({len(common_ports)} common ports)"
    )

    scanned_count = 0
    for port in common_ports:
        src_port = random.randint(1024, 65535)
        pkt = IP(dst=target_ip) / TCP(sport=src_port, dport=port, flags="S")
        send(pkt, verbose=0)

        scanned_count += 1
        print(
            f"  [QUICK SCAN] Probed Port {port} ({scanned_count}/{len(common_ports)})"
        )

        if delay > 0:
            time.sleep(delay)

    print(f"\n[✔] Quick Port Scan completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Port Scan Simulation Tool for NIDS Testing"
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target IP address (e.g., 127.0.0.1 or victim VPN IP)",
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=1,
        help="Start port range (default: 1)",
    )
    parser.add_argument(
        "--end-port",
        type=int,
        default=100,
        help="End port range (default: 100)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Scan top common service ports only",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.01,
        help="Delay between packet probes in seconds (default: 0.01)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("     XAI-NIDS Module 6: Port Scan Attack Simulator")
    print("=" * 60)

    try:
        if args.quick:
            run_quick_scan(args.target, args.delay)
        else:
            run_syn_port_scan(
                args.target, args.start_port, args.end_port, args.delay
            )
    except KeyboardInterrupt:
        print("\n[!] Scan aborted by user.")
    except Exception as exc:
        print(f"\n[✘] Error: {exc}")


if __name__ == "__main__":
    main()