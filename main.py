"""Top-level entry point for XAI-NIDS."""

import sys


def print_usage() -> None:
    print(
        """
XAI-NIDS — Explainable Network Intrusion Detection System
==========================================================

Usage:
  python main.py capture                         Start live IDS capture
  python main.py capture --interface WiFi       Capture from WiFi
  python main.py capture --help                  Show capture options
  python main.py dashboard                       Start the Streamlit dashboard
  python main.py server                          Start the communication server
  python main.py client                          Start the communication client

Live IDS flow:
  Packet capture -> flow generation -> 78 features -> Random Forest
  -> behavioural Port Scan / SYN Flood detection -> SHAP -> dashboard

The dashboard reads the shared state written by the live capture process.
Run it in a second terminal while `python main.py capture` is running.
"""
    )


def main() -> None:
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    if command == "server":
        from communication.server import main as server_main
        server_main()

    elif command == "client":
        from communication.client import main as client_main
        client_main()

    elif command == "capture":
        from capture.capture_daemon import main as capture_main
        raise SystemExit(capture_main(sys.argv[2:]))

    elif command == "dashboard":
        # Streamlit is normally started with `streamlit run dashboard/dashboard.py`.
        # Keep this command as a helpful wrapper without importing Streamlit into
        # the capture process.
        import subprocess
        from pathlib import Path

        dashboard_path = Path(__file__).resolve().parent / "dashboard" / "dashboard.py"
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(dashboard_path)]))

    else:
        print(f"Unknown command: {command!r}")
        print_usage()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
