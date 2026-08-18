# dashboard_bridge.py

import json
import os
import sqlite3
import threading
from typing import Any, Dict, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "ids_events.db")
DEFAULT_STATE_PATH = os.path.join(BASE_DIR, "live_state.json")


class DashboardBridge:
    """Inter-process state bridge using shared JSON state and SQLite history."""

    def __init__(self, max_live_history: int = 100, db_path: str = DEFAULT_DB_PATH, state_path: str = DEFAULT_STATE_PATH):
        self._lock = threading.Lock()
        self._max_history = max_live_history
        self._db_path = db_path
        self._state_path = state_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    src_ip TEXT,
                    src_port INTEGER,
                    dst_ip TEXT,
                    dst_port INTEGER,
                    protocol INTEGER,
                    ml_prediction TEXT,
                    behavioural_detection TEXT,
                    final_decision TEXT,
                    detection_reason TEXT,
                    confidence REAL,
                    top_shap_features TEXT
                )
                """
            )
            existing = {row[1] for row in conn.execute("PRAGMA table_info(detections)").fetchall()}
            migrations = {
                "ml_prediction": "TEXT",
                "behavioural_detection": "TEXT",
                "final_decision": "TEXT",
                "detection_reason": "TEXT",
            }
            for column, column_type in migrations.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE detections ADD COLUMN {column} {column_type}")
            conn.commit()

    def _empty_state(self) -> Dict[str, Any]:
        return {"total_packets": 0, "active_flows": 0, "recent_flows": [], "alerts": []}

    def _read_state(self) -> Dict[str, Any]:
        if not os.path.exists(self._state_path):
            return self._empty_state()
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            base = self._empty_state()
            base.update(state if isinstance(state, dict) else {})
            return base
        except Exception:
            return self._empty_state()

    def _write_state(self, state: Dict[str, Any]) -> None:
        try:
            temp_path = f"{self._state_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(temp_path, self._state_path)
        except Exception as exc:
            print(f"[BRIDGE ERROR] State write failed: {exc}")

    def push_prediction(
        self,
        flow_data: Dict[str, Any],
        prediction: str,
        confidence: float,
        shap_explanation: Optional[Dict[str, float]] = None,
        ml_prediction: Optional[str] = None,
        behavioural_detection: Optional[str] = None,
        final_decision: Optional[str] = None,
        detection_reason: Optional[str] = None,
        port_scan_result: Optional[Dict[str, Any]] = None,
        dos_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        final_decision = final_decision or prediction
        ml_prediction = ml_prediction or prediction
        behavioural_detection = behavioural_detection or "Benign"
        record = {
            "timestamp": flow_data.get("timestamp", flow_data.get("capture_ts", "")),
            "src_ip": flow_data.get("src_ip", "0.0.0.0"),
            "src_port": flow_data.get("src_port", 0),
            "dst_ip": flow_data.get("dst_ip", "0.0.0.0"),
            "dst_port": flow_data.get("dst_port", 0),
            "protocol": flow_data.get("protocol", 0),
            "flow_duration": flow_data.get("Flow Duration", 0),
            "prediction": final_decision,
            "ml_prediction": ml_prediction,
            "behavioural_detection": behavioural_detection,
            "final_decision": final_decision,
            "detection_reason": detection_reason or "Random Forest prediction",
            "confidence": confidence,
            "shap_explanation": shap_explanation or {},
            "port_scan_result": port_scan_result,
            "dos_result": dos_result,
            "raw_features": flow_data,
        }

        with self._lock:
            state = self._read_state()
            recent_flows = state.get("recent_flows", [])
            recent_flows.insert(0, record)
            state["recent_flows"] = recent_flows[: self._max_history]

            if final_decision.upper() != "BENIGN":
                alerts = state.get("alerts", [])
                alerts.insert(0, record)
                state["alerts"] = alerts[:50]
                self._persist_alert(record)

            self._write_state(state)

    def _persist_alert(self, record: Dict[str, Any]) -> None:
        try:
            top_features = json.dumps(dict(list(record["shap_explanation"].items())[:5]))
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO detections
                    (timestamp, src_ip, src_port, dst_ip, dst_port, protocol,
                     ml_prediction, behavioural_detection, final_decision,
                     detection_reason, confidence, top_shap_features)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["timestamp"], record["src_ip"], record["src_port"],
                        record["dst_ip"], record["dst_port"], record["protocol"],
                        record["ml_prediction"], record["behavioural_detection"],
                        record["final_decision"], record["detection_reason"],
                        record["confidence"], top_features,
                    ),
                )
                conn.commit()
        except Exception as exc:
            print(f"[BRIDGE ERROR] SQLite write failed: {exc}")

    def update_telemetry(self, packet_count: int, active_flows: int) -> None:
        with self._lock:
            state = self._read_state()
            state["total_packets"] = packet_count
            state["active_flows"] = active_flows
            self._write_state(state)

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._read_state()

    def check_vpn_status(self) -> bool:
        try:
            import psutil
            return any(name.lower().startswith(("tun", "tap", "wg")) for name in psutil.net_if_addrs())
        except Exception:
            return False


bridge_instance = DashboardBridge()
