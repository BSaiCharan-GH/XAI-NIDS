from collections import defaultdict, deque
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"


class PortScanDetector:
    def __init__(self, window_seconds=5.0, minimum_unique_ports=10, cooldown_seconds=10.0):
        self.window_seconds = float(window_seconds)
        self.minimum_unique_ports = int(minimum_unique_ports)
        self.cooldown_seconds = float(cooldown_seconds)
        self.connections = defaultdict(deque)
        self.last_alert = {}

    @staticmethod
    def _timestamp(features):
        value = features.get("capture_ts")
        if value is not None:
            try:
                return pd.to_datetime(value).timestamp()
            except Exception:
                pass
        return time.time()

    def process_flow(self, features):
        src_ip = features.get("src_ip")
        dst_ip = features.get("dst_ip")
        dst_port = features.get("dst_port")
        if not src_ip or not dst_ip or dst_port is None:
            return None
        try:
            dst_port = int(dst_port)
        except (TypeError, ValueError):
            return None

        now = self._timestamp(features)
        key = (str(src_ip), str(dst_ip))
        self.connections[key].append({"time": now, "port": dst_port})
        cutoff = now - self.window_seconds
        while self.connections[key] and self.connections[key][0]["time"] < cutoff:
            self.connections[key].popleft()

        flows = self.connections[key]
        unique_ports = len({item["port"] for item in flows})
        connections = len(flows)
        elapsed = max(now - flows[0]["time"], 0.001) if flows else 0.001
        scan_rate = connections / elapsed
        detected = unique_ports >= self.minimum_unique_ports
        suppressed = False

        if detected:
            last = self.last_alert.get(key, 0.0)
            if now - last < self.cooldown_seconds:
                suppressed = True
            else:
                self.last_alert[key] = now

        return {
            "detected": detected,
            "source": src_ip,
            "destination": dst_ip,
            "unique_ports": unique_ports,
            "connections": connections,
            "scan_rate": scan_rate,
            "suppressed": suppressed,
        }


class SYNFloodDetector:
    def __init__(self, window_seconds=5.0, minimum_syn_packets=20, minimum_syn_flows=10,
                 minimum_syn_rate=10.0, maximum_ack_ratio=0.25, cooldown_seconds=10.0):
        self.window_seconds = float(window_seconds)
        self.minimum_syn_packets = int(minimum_syn_packets)
        self.minimum_syn_flows = int(minimum_syn_flows)
        self.minimum_syn_rate = float(minimum_syn_rate)
        self.maximum_ack_ratio = float(maximum_ack_ratio)
        self.cooldown_seconds = float(cooldown_seconds)
        self.connections = defaultdict(deque)
        self.last_alert = {}

    @staticmethod
    def _number(features, name, default=0.0):
        try:
            value = float(features.get(name, default))
            return value if np.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _timestamp(features):
        value = features.get("capture_ts")
        if value is not None:
            try:
                return pd.to_datetime(value).timestamp()
            except Exception:
                pass
        return time.time()

    def process_flow(self, features):
        src_ip = features.get("src_ip")
        dst_ip = features.get("dst_ip")
        if not src_ip or not dst_ip:
            return None

        try:
            protocol = int(features.get("protocol", 0))
        except (TypeError, ValueError):
            protocol = 0
        if protocol != 6:
            return None

        syn = self._number(features, "SYN Flag Count")
        ack = self._number(features, "ACK Flag Count")
        rst = self._number(features, "RST Flag Count")
        fwd = self._number(features, "Total Fwd Packets")
        bwd = self._number(features, "Total Backward Packets")
        if syn <= 0:
            return None

        now = self._timestamp(features)
        key = (str(src_ip), str(dst_ip))
        self.connections[key].append({"time": now, "syn": syn, "ack": ack, "rst": rst, "fwd": fwd, "bwd": bwd})
        cutoff = now - self.window_seconds
        while self.connections[key] and self.connections[key][0]["time"] < cutoff:
            self.connections[key].popleft()

        observations = self.connections[key]
        total_syn = sum(x["syn"] for x in observations)
        total_ack = sum(x["ack"] for x in observations)
        total_rst = sum(x["rst"] for x in observations)
        total_fwd = sum(x["fwd"] for x in observations)
        total_bwd = sum(x["bwd"] for x in observations)
        syn_flows = len(observations)
        elapsed = min(self.window_seconds, max(now - observations[0]["time"], 0.1))
        syn_rate = total_syn / elapsed
        ack_ratio = total_ack / max(total_syn + total_ack, 1.0)
        syn_ratio = total_syn / max(total_syn + total_ack, 1.0)
        one_way = sum(1 for x in observations if x["syn"] > 0 and x["bwd"] <= 0)
        one_way_ratio = one_way / max(syn_flows, 1)

        detected = (
            total_syn >= self.minimum_syn_packets
            and syn_flows >= self.minimum_syn_flows
            and syn_rate >= self.minimum_syn_rate
            and ack_ratio <= self.maximum_ack_ratio
            and one_way_ratio >= 0.70
        )
        suppressed = False
        if detected:
            last = self.last_alert.get(key, 0.0)
            if now - last < self.cooldown_seconds:
                suppressed = True
            else:
                self.last_alert[key] = now

        return {
            "detected": detected,
            "source": src_ip,
            "destination": dst_ip,
            "syn_packets": total_syn,
            "syn_flows": syn_flows,
            "connections": syn_flows,
            "syn_rate": syn_rate,
            "syn_ratio": syn_ratio,
            "ack_packets": total_ack,
            "ack_ratio": ack_ratio,
            "rst_packets": total_rst,
            "one_way_ratio": one_way_ratio,
            "fwd_packets": total_fwd,
            "bwd_packets": total_bwd,
            "suppressed": suppressed,
        }


class LiveInference:
    def __init__(self, model_path=None, dashboard_bridge=None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path
        self.dashboard_bridge = dashboard_bridge

        print("\n" + "=" * 60)
        print("XAI-IDS — Live Machine Learning Inference")
        print("=" * 60)
        print(f"[ML] Loading model: {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        bundle = joblib.load(self.model_path)
        if isinstance(bundle, dict) and "model" in bundle:
            self.model = bundle["model"]
            self.features = list(bundle.get("features", getattr(self.model, "feature_names_in_", [])))
        else:
            self.model = bundle
            self.features = list(getattr(self.model, "feature_names_in_", []))

        if not self.features:
            raise ValueError("The trained model does not contain a feature list.")
        if len(self.features) != 78:
            raise ValueError(f"Expected 78 model features, found {len(self.features)}")

        print(f"[ML] Model type: {type(self.model).__name__}")
        print(f"[ML] Features: {len(self.features)}")
        print(f"[ML] Classes: {list(self.model.classes_)}")

        self.explainer = None
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.model)
            print("[XAI] SHAP TreeExplainer initialized.")
        except Exception as exc:
            print(f"[XAI] SHAP unavailable: {exc}")

        self.port_scan_detector = PortScanDetector()
        self.syn_flood_detector = SYNFloodDetector()
        print("[SCAN] Port Scan: 10 unique destination ports in 5 seconds")
        print("[DOS] SYN Flood: 20 SYN packets + 10 flows + low ACK ratio in 5 seconds")
        print("[ML] Live inference ready.\n" + "=" * 60 + "\n")

    def _build_input(self, features):
        row = {}
        for feature in self.features:
            value = features.get(feature, 0.0)
            try:
                value = float(value)
                if not np.isfinite(value):
                    value = 0.0
            except (TypeError, ValueError):
                value = 0.0
            row[feature] = value
        return pd.DataFrame([row], columns=self.features)

    def _confidence(self, X, prediction):
        try:
            probabilities = self.model.predict_proba(X)[0]
            classes = list(self.model.classes_)
            return float(probabilities[classes.index(prediction)])
        except Exception:
            return 0.0

    def _shap_values(self, X, prediction):
        if self.explainer is None:
            return {}
        try:
            raw = self.explainer.shap_values(X)
            classes = list(self.model.classes_)
            class_index = classes.index(prediction)
            values = np.asarray(raw)

            if isinstance(raw, list):
                values = np.asarray(raw[class_index])[0]
            elif values.ndim == 3:
                if values.shape[0] == 1:
                    values = values[0, :, class_index]
                else:
                    values = values[:, :, class_index][0]
            elif values.ndim == 2:
                values = values[0]
            else:
                values = values.reshape(-1)

            result = {name: float(value) for name, value in zip(self.features, values)}
            return dict(sorted(result.items(), key=lambda item: abs(item[1]), reverse=True)[:10])
        except Exception as exc:
            print(f"[XAI] Explanation failed: {exc}")
            return {}

    @staticmethod
    def _behaviour_label(port_result, dos_result):
        if dos_result and dos_result.get("detected") and not dos_result.get("suppressed"):
            return "DoS"
        if port_result and port_result.get("detected") and not port_result.get("suppressed"):
            return "Port Scan"
        return "Benign"

    def process_flow(self, features):
        try:
            X = self._build_input(features)
            ml_prediction = str(self.model.predict(X)[0])
            ml_confidence = self._confidence(X, ml_prediction)
            shap_explanation = self._shap_values(X, ml_prediction)

            port_result = self.port_scan_detector.process_flow(features)
            dos_result = self.syn_flood_detector.process_flow(features)
            behavioural = self._behaviour_label(port_result, dos_result)

            if behavioural == "DoS":
                final_decision = "DoS"
                reason = "SYN Flood behavioural detector"
            elif behavioural == "Port Scan":
                final_decision = "Port Scan"
                reason = "Port Scan behavioural detector"
            else:
                final_decision = ml_prediction
                reason = "Random Forest prediction"

            print(f"\n[ML] Prediction           : {ml_prediction}")
            print(f"[ML] Confidence           : {ml_confidence * 100:.2f}%")
            print(f"[BEHAVIOUR] Detection     : {behavioural}")
            print(f"[SECURITY] Final Decision : {final_decision}")
            print(f"[SECURITY] Reason         : {reason}")

            if port_result and port_result.get("detected") and not port_result.get("suppressed"):
                print("[ALERT] PORT SCAN DETECTED")
                print(f"        Source           : {port_result['source']}")
                print(f"        Destination      : {port_result['destination']}")
                print(f"        Unique ports     : {port_result['unique_ports']}")
                print(f"        Connections      : {port_result['connections']}")
                print(f"        Scan rate        : {port_result['scan_rate']:.2f} flows/sec")

            if dos_result and dos_result.get("detected") and not dos_result.get("suppressed"):
                print("[ALERT] SYN FLOOD / DoS DETECTED")
                print(f"        Source           : {dos_result['source']}")
                print(f"        Destination      : {dos_result['destination']}")
                print(f"        SYN packets      : {dos_result['syn_packets']:.0f}")
                print(f"        SYN flows        : {dos_result['syn_flows']}")
                print(f"        SYN rate         : {dos_result['syn_rate']:.2f} SYN/sec")
                print(f"        SYN ratio        : {dos_result['syn_ratio'] * 100:.2f}%")
                print(f"        ACK packets      : {dos_result['ack_packets']:.0f}")
                print(f"        ACK ratio        : {dos_result['ack_ratio'] * 100:.2f}%")

            if shap_explanation:
                print("[XAI] Top contributing features:")
                for name, value in list(shap_explanation.items())[:5]:
                    print(f"       {name:<32} {value:+.4f}")

            if self.dashboard_bridge is not None:
                self.dashboard_bridge.push_prediction(
                    flow_data=features,
                    prediction=final_decision,
                    confidence=ml_confidence,
                    shap_explanation=shap_explanation,
                    ml_prediction=ml_prediction,
                    behavioural_detection=behavioural,
                    final_decision=final_decision,
                    detection_reason=reason,
                    port_scan_result=port_result,
                    dos_result=dos_result,
                )
                print("[DASHBOARD] Live prediction pushed.")

            return {
                "ml_prediction": ml_prediction,
                "ml_confidence": ml_confidence,
                "behavioural_detection": behavioural,
                "final_decision": final_decision,
                "detection_reason": reason,
                "port_scan": port_result,
                "dos": dos_result,
                "shap_explanation": shap_explanation,
            }
        except Exception as exc:
            print(f"[ML ERROR] Live inference failed: {exc}")
            return None


if __name__ == "__main__":
    print("This module defines the live inference engine.")
    print("Start the complete IDS with:")
    print("  python main.py capture")
