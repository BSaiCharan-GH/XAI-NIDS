# dashboard_bridge.py

import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

# Resolve absolute path to the directory where dashboard_bridge.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "ids_events.db")
DEFAULT_STATE_PATH = os.path.join(BASE_DIR, "live_state.json")


class DashboardBridge:
    """Inter-process state bridge using shared file-backed storage and SQLite."""

    def __init__(
        self,
        max_live_history: int = 100,
        db_path: str = DEFAULT_DB_PATH,
        state_path: str = DEFAULT_STATE_PATH,
    ):
        self._lock = threading.Lock()
        self._max_history = max_live_history
        self._db_path = db_path
        self._state_path = state_path

        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for historical log persistence."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    src_ip TEXT,
                    src_port INTEGER,
                    dst_ip TEXT,
                    dst_port INTEGER,
                    protocol INTEGER,
                    prediction TEXT,
                    confidence REAL,
                    top_shap_features TEXT
                )
            """
            )
            conn.commit()

    def _read_state(self) -> Dict[str, Any]:
        """Read the shared state file across processes."""
        if not os.path.exists(self._state_path):
            return {
                "total_packets": 0,
                "active_flows": 0,
                "recent_flows": [],
                "alerts": [],
            }
        try:
            with open(self._state_path, "r") as f:
                return json.load(f)
        except Exception:
            return {
                "total_packets": 0,
                "active_flows": 0,
                "recent_flows": [],
                "alerts": [],
            }

    def _write_state(self, state: Dict[str, Any]) -> None:
        """Write the shared state file across processes safely."""
        try:
            temp_path = f"{self._state_path}.tmp"
            with open(temp_path, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(temp_path, self._state_path)
        except Exception as exc:
            print(f"[BRIDGE ERROR] State write failed: {exc}")

    def push_prediction(
        self,
        flow_data: Dict[str, Any],
        prediction: str,
        confidence: float,
        shap_explanation: Optional[Dict[str, float]] = None,
    ) -> None:
        """Receive ML/Mock output and update shared live state + SQLite logs."""
        record = {
            "timestamp": flow_data.get(
                "timestamp", flow_data.get("capture_ts", "")
            ),
            "src_ip": flow_data.get("src_ip", "0.0.0.0"),
            "src_port": flow_data.get("src_port", 0),
            "dst_ip": flow_data.get("dst_ip", "0.0.0.0"),
            "dst_port": flow_data.get("dst_port", 0),
            "protocol": flow_data.get("protocol", 0),
            "flow_duration": flow_data.get("Flow Duration", 0),
            "prediction": prediction,
            "confidence": confidence,
            "shap_explanation": shap_explanation or {},
            "raw_features": flow_data,
        }

        with self._lock:
            state = self._read_state()

            # Insert flow at beginning of recent flows queue
            recent_flows = state.get("recent_flows", [])
            recent_flows.insert(0, record)
            state["recent_flows"] = recent_flows[: self._max_history]

            if prediction.upper() != "BENIGN":
                alerts = state.get("alerts", [])
                alerts.insert(0, record)
                state["alerts"] = alerts[:50]
                self._persist_alert(record)

            self._write_state(state)

    def _persist_alert(self, record: Dict[str, Any]) -> None:
        """Write threat detections to SQLite database standardizing JSON serialization."""
        try:
            top_features_str = json.dumps(
                dict(list(record["shap_explanation"].items())[:5])
            )
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO detections 
                    (timestamp, src_ip, src_port, dst_ip, dst_port, protocol, prediction, confidence, top_shap_features)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record["timestamp"],
                        record["src_ip"],
                        record["src_port"],
                        record["dst_ip"],
                        record["dst_port"],
                        record["protocol"],
                        record["prediction"],
                        record["confidence"],
                        top_features_str,
                    ),
                )
                conn.commit()
        except Exception as exc:
            print(f"[BRIDGE ERROR] SQLite write failed: {exc}")

    def update_telemetry(self, packet_count: int, active_flows: int) -> None:
        """Update live telemetry counters."""
        with self._lock:
            state = self._read_state()
            state["total_packets"] = packet_count
            state["active_flows"] = active_flows
            self._write_state(state)

    def get_snapshot(self) -> Dict[str, Any]:
        """Fetch snapshot of current system state for UI rendering."""
        with self._lock:
            return self._read_state()

    def check_vpn_status(self) -> bool:
        """Check if the designated VPN network interface is active."""
        try:
            import psutil

            interfaces = psutil.net_if_addrs()
            return "tun0" in interfaces  # Adjust if VPN interface name differs
        except Exception:
            return False


# Shared Singleton Instance
bridge_instance = DashboardBridge()