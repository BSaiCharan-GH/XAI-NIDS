import argparse
import queue
import signal
import sys
import threading

from capture.constants import (
    CAPTURE_CSV_DIR,
    DEFAULT_FILTER,
    FLOW_TIMEOUT,
)

from capture.flow_tracker import FlowTracker
from capture.capture_logger import CaptureLogger
from capture.sniffer import PacketSniffer
from ml.live_inference import LiveInference
from dashboard.dashboard_bridge import bridge_instance


stop_event = threading.Event()


def handle_signal(signum, frame):
    print("\n[INFO] Shutdown signal received.")
    stop_event.set()


def run_expiry_thread(tracker):
    interval = max(10.0, FLOW_TIMEOUT / 10)

    while not stop_event.is_set():
        try:
            expired = tracker.expire_idle_flows()

            if expired:
                print(f"[EXPIRY] {expired} idle flow(s) expired.")

        except Exception as exc:
            print(f"[ERROR] Flow expiry error: {exc}")

        stop_event.wait(interval)


def update_dashboard_telemetry(tracker):
    """
    Publish packet and active-flow counters to the shared dashboard state.

    This is intentionally done from the daemon's main loop rather than
    once per packet so that file-backed dashboard state is not rewritten
    for every captured packet.
    """
    try:
        stats = tracker.stats
        bridge_instance.update_telemetry(
            packet_count=int(stats.get("packets_seen", 0)),
            active_flows=int(tracker.active_flow_count),
        )
    except Exception as exc:
        print(f"[DASHBOARD ERROR] Telemetry update failed: {exc}")


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="XAI-IDS Network Packet Capture and Feature Extraction"
    )

    parser.add_argument(
        "--interface",
        "-i",
        default=None,
        help="Network interface to capture from"
    )

    parser.add_argument(
        "--output",
        "-o",
        default=CAPTURE_CSV_DIR,
        help=f"Output directory for CSV files (default: {CAPTURE_CSV_DIR})"
    )

    parser.add_argument(
        "--filter",
        "-f",
        default=DEFAULT_FILTER,
        help=f"BPF capture filter (default: {DEFAULT_FILTER})"
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=FLOW_TIMEOUT,
        help=f"Flow idle timeout in seconds (default: {FLOW_TIMEOUT})"
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-flow console output"
    )

    args = parser.parse_args(argv)

    print()
    print("=" * 60)
    print("XAI-IDS — Network Packet Capture")
    print("=" * 60)
    print()

    print("[INFO] Configuration:")
    print(f"       Interface : {args.interface or 'default'}")
    print(f"       Filter    : {args.filter}")
    print(f"       Timeout   : {args.timeout} seconds")
    print(f"       Output    : {args.output}")
    print()

    # ---------------------------------------------------------------
    # Shared queue between FlowTracker and CaptureLogger
    # ---------------------------------------------------------------

    completed_queue = queue.Queue()

    # ---------------------------------------------------------------
    # Dashboard bridge
    # ---------------------------------------------------------------

    print("[OK] Dashboard bridge initialized.")

    # ---------------------------------------------------------------
    # Machine learning model
    # ---------------------------------------------------------------

    try:
        inference = LiveInference(
            dashboard_bridge=bridge_instance
        )

        print("[OK] Machine learning inference initialized.")

    except Exception as exc:
        print(f"[ERROR] Failed to load ML model: {exc}")
        return 1

    # ---------------------------------------------------------------
    # Flow tracker
    # ---------------------------------------------------------------

    try:
        tracker = FlowTracker(
            completed_queue=completed_queue,
            flow_timeout=args.timeout
        )

        print("[OK] Flow tracker initialized.")

    except Exception as exc:
        print(f"[ERROR] Failed to initialize FlowTracker: {exc}")
        return 1

    # ---------------------------------------------------------------
    # CSV logger
    # ---------------------------------------------------------------

    try:
        logger = CaptureLogger(
            completed_queue=completed_queue,
            output_dir=args.output,
            on_flow_completed=inference.process_flow,
            verbose=not args.quiet
        )

        print("[OK] Capture logger initialized.")
        print(f"[OK] Writing flows to: {logger.csv_path}")

    except Exception as exc:
        print(f"[ERROR] Failed to initialize CaptureLogger: {exc}")
        return 1

    # ---------------------------------------------------------------
    # Packet sniffer
    # ---------------------------------------------------------------

    try:
        sniffer = PacketSniffer(
            flow_tracker=tracker,
            interface=args.interface,
            bpf_filter=args.filter
        )

        print("[OK] Packet sniffer initialized.")

    except Exception as exc:
        print(f"[ERROR] Failed to initialize PacketSniffer: {exc}")

        try:
            logger.stop()
        except Exception:
            pass

        return 1

    # ---------------------------------------------------------------
    # Signal handlers
    # ---------------------------------------------------------------

    signal.signal(signal.SIGINT, handle_signal)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    # ---------------------------------------------------------------
    # Start logger
    # ---------------------------------------------------------------

    try:
        logger.start()
        print("[OK] Capture logger started.")

    except Exception as exc:
        print(f"[ERROR] Failed to start logger: {exc}")
        return 1

    # ---------------------------------------------------------------
    # Start flow expiry thread
    # ---------------------------------------------------------------

    expiry_thread = threading.Thread(
        target=run_expiry_thread,
        args=(tracker,),
        name="flow-expiry",
        daemon=True
    )

    expiry_thread.start()

    print("[OK] Flow expiry thread started.")
    print()

    # ---------------------------------------------------------------
    # Start capture
    # ---------------------------------------------------------------

    try:

        print("=" * 60)
        print("CAPTURE STARTED")
        print("=" * 60)
        print()
        print("Listening for network traffic...")
        print("Press Ctrl+C to stop.")
        print()

        sniffer.start()

        # Keep the main thread alive while the sniffer captures packets.
        # Also publish dashboard telemetry periodically.
        while not stop_event.wait(0.5):
            update_dashboard_telemetry(tracker)

    except KeyboardInterrupt:

        print("\n[INFO] Ctrl+C received.")

    except Exception as exc:

        print(f"\n[ERROR] Packet capture failed: {exc}")

    finally:

        stop_event.set()

        print()
        print("=" * 60)
        print("STOPPING CAPTURE")
        print("=" * 60)

        # -----------------------------------------------------------
        # Stop sniffer
        # -----------------------------------------------------------

        try:
            sniffer.stop()
            print("[OK] Packet sniffer stopped.")
        except Exception as exc:
            print(f"[WARNING] Error stopping sniffer: {exc}")

        # -----------------------------------------------------------
        # Flush active flows
        # -----------------------------------------------------------

        try:
            active = tracker.active_flow_count

            if active:
                print(f"[INFO] Flushing {active} active flow(s)...")

            tracker.flush_all()

        except Exception as exc:
            print(f"[WARNING] Error flushing flows: {exc}")

        # -----------------------------------------------------------
        # Allow the logger to drain the completed-flow queue.
        # -----------------------------------------------------------

        try:
            logger.stop()
            print("[OK] Capture logger stopped.")
        except Exception as exc:
            print(f"[WARNING] Error stopping logger: {exc}")

        # -----------------------------------------------------------
        # Final dashboard telemetry
        # -----------------------------------------------------------

        update_dashboard_telemetry(tracker)

        # -----------------------------------------------------------
        # Wait for expiry thread
        # -----------------------------------------------------------

        expiry_thread.join(timeout=2.0)

        print()
        print("Capture stopped.")

        try:
            print(f"Output CSV: {logger.csv_path}")
        except Exception:
            pass

        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
