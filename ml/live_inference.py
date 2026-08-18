from collections import defaultdict, deque
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"


def capture_timestamp(value):
    """
    Convert a capture timestamp into Unix seconds.

    Numeric timestamps are already Unix seconds and must not be passed
    through pd.to_datetime(), because pandas interprets bare integers
    as nanoseconds by default.
    """

    if value is None:
        return time.time()

    if isinstance(value, (int, float, np.integer, np.floating)):

        value = float(value)

        if np.isfinite(value):
            return value

        return time.time()

    try:
        return pd.Timestamp(value).timestamp()

    except Exception:
        return time.time()


class PortScanDetector:

    def __init__(
        self,
        window_seconds=5.0,
        minimum_unique_ports=10,
        cooldown_seconds=10.0
    ):

        self.window_seconds = float(window_seconds)
        self.minimum_unique_ports = int(minimum_unique_ports)
        self.cooldown_seconds = float(cooldown_seconds)

        self.connections = defaultdict(deque)
        self.last_alert = {}

    def process_flow(self, features):

        source = features.get("src_ip")
        destination = features.get("dst_ip")
        destination_port = features.get("dst_port")

        if not source or not destination:
            return None

        if destination_port is None:
            return None

        try:
            destination_port = int(destination_port)

        except (TypeError, ValueError):
            return None

        now = capture_timestamp(
            features.get("capture_ts")
        )

        key = (
            str(source),
            str(destination)
        )

        self.connections[key].append(
            {
                "time": now,
                "port": destination_port
            }
        )

        cutoff = now - self.window_seconds

        while (
            self.connections[key]
            and
            self.connections[key][0]["time"] < cutoff
        ):

            self.connections[key].popleft()

        flows = self.connections[key]

        unique_ports = len(
            {
                item["port"]
                for item in flows
            }
        )

        connections = len(flows)

        if flows:

            elapsed = max(
                now - flows[0]["time"],
                0.001
            )

        else:

            elapsed = 0.001

        scan_rate = (
            connections /
            elapsed
        )

        detected = (
            unique_ports >=
            self.minimum_unique_ports
        )

        suppressed = False

        if detected:

            last_alert = self.last_alert.get(
                key,
                0.0
            )

            if (
                now - last_alert
                < self.cooldown_seconds
            ):

                suppressed = True

            else:

                self.last_alert[key] = now

        return {
            "detected": detected,
            "source": source,
            "destination": destination,
            "unique_ports": unique_ports,
            "connections": connections,
            "scan_rate": scan_rate,
            "suppressed": suppressed
        }


class SYNFloodDetector:

    def __init__(
        self,
        window_seconds=5.0,
        minimum_syn_packets=20,
        minimum_syn_flows=10,
        minimum_syn_rate=10.0,
        maximum_ack_ratio=0.25,
        cooldown_seconds=10.0
    ):

        self.window_seconds = float(
            window_seconds
        )

        self.minimum_syn_packets = int(
            minimum_syn_packets
        )

        self.minimum_syn_flows = int(
            minimum_syn_flows
        )

        self.minimum_syn_rate = float(
            minimum_syn_rate
        )

        self.maximum_ack_ratio = float(
            maximum_ack_ratio
        )

        self.cooldown_seconds = float(
            cooldown_seconds
        )

        self.connections = defaultdict(
            deque
        )

        self.last_alert = {}

    @staticmethod
    def number(
        features,
        name,
        default=0.0
    ):

        try:

            value = float(
                features.get(
                    name,
                    default
                )
            )

            if np.isfinite(value):
                return value

        except (
            TypeError,
            ValueError
        ):

            pass

        return default

    def process_flow(self, features):

        source = features.get("src_ip")
        destination = features.get("dst_ip")

        if not source or not destination:
            return None

        try:

            protocol = int(
                features.get(
                    "protocol",
                    0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            protocol = 0

        # TCP only
        if protocol != 6:
            return None

        syn = self.number(
            features,
            "SYN Flag Count"
        )

        ack = self.number(
            features,
            "ACK Flag Count"
        )

        rst = self.number(
            features,
            "RST Flag Count"
        )

        fwd_packets = self.number(
            features,
            "Total Fwd Packets"
        )

        bwd_packets = self.number(
            features,
            "Total Backward Packets"
        )

        if syn <= 0:
            return None

        now = capture_timestamp(
            features.get("capture_ts")
        )

        key = (
            str(source),
            str(destination)
        )

        self.connections[key].append(
            {
                "time": now,
                "syn": syn,
                "ack": ack,
                "rst": rst,
                "fwd": fwd_packets,
                "bwd": bwd_packets
            }
        )

        cutoff = (
            now -
            self.window_seconds
        )

        while (
            self.connections[key]
            and
            self.connections[key][0]["time"]
            < cutoff
        ):

            self.connections[key].popleft()

        observations = self.connections[key]

        total_syn = sum(
            item["syn"]
            for item in observations
        )

        total_ack = sum(
            item["ack"]
            for item in observations
        )

        total_rst = sum(
            item["rst"]
            for item in observations
        )

        total_fwd = sum(
            item["fwd"]
            for item in observations
        )

        total_bwd = sum(
            item["bwd"]
            for item in observations
        )

        syn_flows = len(
            observations
        )

        if observations:

            elapsed = max(
                now -
                observations[0]["time"],
                0.1
            )

        else:

            elapsed = 0.1

        elapsed = min(
            self.window_seconds,
            elapsed
        )

        syn_rate = (
            total_syn /
            elapsed
        )

        ack_ratio = (
            total_ack /
            max(
                total_syn +
                total_ack,
                1.0
            )
        )

        syn_ratio = (
            total_syn /
            max(
                total_syn +
                total_ack,
                1.0
            )
        )

        one_way_connections = sum(
            1
            for item in observations
            if (
                item["syn"] > 0
                and
                item["bwd"] <= 0
            )
        )

        one_way_ratio = (
            one_way_connections /
            max(
                syn_flows,
                1
            )
        )

        detected = (
            total_syn >=
            self.minimum_syn_packets
            and
            syn_flows >=
            self.minimum_syn_flows
            and
            syn_rate >=
            self.minimum_syn_rate
            and
            ack_ratio <=
            self.maximum_ack_ratio
            and
            one_way_ratio >= 0.70
        )

        suppressed = False

        if detected:

            last_alert = self.last_alert.get(
                key,
                0.0
            )

            if (
                now - last_alert
                <
                self.cooldown_seconds
            ):

                suppressed = True

            else:

                self.last_alert[key] = now

        return {
            "detected": detected,
            "source": source,
            "destination": destination,
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
            "suppressed": suppressed
        }


class LiveInference:

    def __init__(
        self,
        model_path=None,
        dashboard_bridge=None
    ):

        self.model_path = (
            Path(model_path)
            if model_path
            else DEFAULT_MODEL_PATH
        )

        if not self.model_path.is_absolute():

            self.model_path = (
                PROJECT_ROOT /
                self.model_path
            )

        print()
        print("=" * 60)
        print(
            "XAI-IDS - Live Machine Learning Inference"
        )
        print("=" * 60)

        print(
            "[ML] Loading model: "
            f"{self.model_path}"
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                "Model not found: "
                f"{self.model_path}"
            )

        bundle = joblib.load(
            self.model_path
        )

        if (
            isinstance(bundle, dict)
            and
            "model" in bundle
        ):

            self.model = bundle["model"]

            self.features = list(
                bundle.get(
                    "features",
                    getattr(
                        self.model,
                        "feature_names_in_",
                        []
                    )
                )
            )

        else:

            self.model = bundle

            self.features = list(
                getattr(
                    self.model,
                    "feature_names_in_",
                    []
                )
            )

        if not self.features:

            raise ValueError(
                "The trained model does not "
                "contain a feature list."
            )

        print(
            "[ML] Model type: "
            f"{type(self.model).__name__}"
        )

        print(
            "[ML] Features: "
            f"{len(self.features)}"
        )

        print(
            "[ML] Classes: "
            f"{list(self.model.classes_)}"
        )

        self.explainer = None

        try:

            import shap

            self.explainer = (
                shap.TreeExplainer(
                    self.model
                )
            )

            print(
                "[XAI] SHAP TreeExplainer initialized."
            )

        except Exception as error:

            print(
                "[XAI] SHAP unavailable: "
                f"{error}"
            )

        self.port_scan_detector = (
            PortScanDetector()
        )

        self.syn_flood_detector = (
            SYNFloodDetector()
        )

        self.dashboard_bridge = (
            dashboard_bridge
        )

        print(
            "[SCAN] Port Scan detector ready."
        )

        print(
            "[DOS] SYN Flood detector ready."
        )

        print(
            "[ML] Live inference ready."
        )

        print("=" * 60)
        print()

    def build_input(self, features):

        row = {}

        for feature in self.features:

            value = features.get(
                feature,
                0.0
            )

            try:

                value = float(value)

                if not np.isfinite(value):
                    value = 0.0

            except (
                TypeError,
                ValueError
            ):

                value = 0.0

            row[feature] = value

        return pd.DataFrame(
            [row],
            columns=self.features
        )

    def confidence(
        self,
        X,
        prediction
    ):

        try:

            probabilities = (
                self.model
                .predict_proba(X)[0]
            )

            classes = list(
                self.model.classes_
            )

            index = classes.index(
                prediction
            )

            return float(
                probabilities[index]
            )

        except Exception:

            return 0.0

    def shap_values(
        self,
        X,
        prediction
    ):

        if self.explainer is None:
            return {}

        try:

            raw = (
                self.explainer
                .shap_values(X)
            )

            classes = list(
                self.model.classes_
            )

            class_index = classes.index(
                prediction
            )

            if isinstance(raw, list):

                values = np.asarray(
                    raw[class_index]
                )[0]

            else:

                values = np.asarray(raw)

                if values.ndim == 3:

                    if values.shape[0] == 1:

                        values = values[
                            0,
                            :,
                            class_index
                        ]

                    else:

                        values = values[
                            :,
                            :,
                            class_index
                        ][0]

                elif values.ndim == 2:

                    values = values[0]

                else:

                    values = values.reshape(
                        -1
                    )

            result = {}

            for name, value in zip(
                self.features,
                values
            ):

                result[name] = float(
                    value
                )

            result = dict(
                sorted(
                    result.items(),
                    key=lambda item:
                    abs(item[1]),
                    reverse=True
                )[:10]
            )

            return result

        except Exception as error:

            print(
                "[XAI] Explanation failed: "
                f"{error}"
            )

            return {}

    @staticmethod
    def behaviour_label(
        port_result,
        dos_result
    ):
        """
        Behavioural detection determines the security decision.

        Important:
        The 'suppressed' flag only prevents repeated alert messages.
        It must NOT turn an already detected attack back into Benign.

        Therefore:
            detected=True, suppressed=True
        still returns the corresponding attack class.
        """

        if (
            dos_result
            and
            dos_result.get("detected")
        ):

            return "DoS"

        if (
            port_result
            and
            port_result.get("detected")
        ):

            return "Port Scan"

        return "Benign"

    def process_flow(self, features):

        try:

            X = self.build_input(
                features
            )

            ml_prediction = str(
                self.model.predict(X)[0]
            )

            ml_confidence = (
                self.confidence(
                    X,
                    ml_prediction
                )
            )

            shap_explanation = (
                self.shap_values(
                    X,
                    ml_prediction
                )
            )

            port_result = (
                self.port_scan_detector
                .process_flow(features)
            )

            dos_result = (
                self.syn_flood_detector
                .process_flow(features)
            )

            behavioural_detection = (
                self.behaviour_label(
                    port_result,
                    dos_result
                )
            )

            if (
                behavioural_detection
                == "DoS"
            ):

                final_decision = "DoS"

                detection_reason = (
                    "SYN Flood behavioural detector"
                )

            elif (
                behavioural_detection
                == "Port Scan"
            ):

                final_decision = "Port Scan"

                detection_reason = (
                    "Port Scan behavioural detector"
                )

            else:

                final_decision = (
                    ml_prediction
                )

                detection_reason = (
                    "Random Forest prediction"
                )

            print()
            print(
                "[ML] Prediction           : "
                f"{ml_prediction}"
            )

            print(
                "[ML] Confidence           : "
                f"{ml_confidence * 100:.2f}%"
            )

            print(
                "[BEHAVIOUR] Detection     : "
                f"{behavioural_detection}"
            )

            print(
                "[SECURITY] Final Decision : "
                f"{final_decision}"
            )

            print(
                "[SECURITY] Reason         : "
                f"{detection_reason}"
            )

            if (
                port_result
                and
                port_result.get("detected")
                and
                not port_result.get("suppressed")
            ):

                print()
                print(
                    "[ALERT] PORT SCAN DETECTED"
                )

                print(
                    "        Source       : "
                    f"{port_result['source']}"
                )

                print(
                    "        Destination  : "
                    f"{port_result['destination']}"
                )

                print(
                    "        Unique ports : "
                    f"{port_result['unique_ports']}"
                )

                print(
                    "        Connections   : "
                    f"{port_result['connections']}"
                )

                print(
                    "        Scan rate    : "
                    f"{port_result['scan_rate']:.2f} "
                    "flows/sec"
                )

            if (
                dos_result
                and
                dos_result.get("detected")
                and
                not dos_result.get("suppressed")
            ):

                print()
                print(
                    "[ALERT] SYN FLOOD / DoS DETECTED"
                )

                print(
                    "        Source       : "
                    f"{dos_result['source']}"
                )

                print(
                    "        Destination  : "
                    f"{dos_result['destination']}"
                )

                print(
                    "        SYN packets  : "
                    f"{dos_result['syn_packets']:.0f}"
                )

                print(
                    "        SYN flows    : "
                    f"{dos_result['syn_flows']}"
                )

                print(
                    "        SYN rate     : "
                    f"{dos_result['syn_rate']:.2f} "
                    "SYN/sec"
                )

                print(
                    "        SYN ratio    : "
                    f"{dos_result['syn_ratio'] * 100:.2f}%"
                )

                print(
                    "        ACK packets  : "
                    f"{dos_result['ack_packets']:.0f}"
                )

                print(
                    "        ACK ratio    : "
                    f"{dos_result['ack_ratio'] * 100:.2f}%"
                )

            if shap_explanation:

                print()
                print(
                    "[XAI] Top contributing features:"
                )

                for name, value in list(
                    shap_explanation.items()
                )[:5]:

                    print(
                        f"       "
                        f"{name:<32}"
                        f" {value:+.4f}"
                    )

            if self.dashboard_bridge:

                try:

                    self.dashboard_bridge.push_prediction(
                        flow_data=features,
                        prediction=final_decision,
                        confidence=ml_confidence,
                        shap_explanation=shap_explanation,
                        ml_prediction=ml_prediction,
                        behavioural_detection=behavioural_detection,
                        final_decision=final_decision,
                        detection_reason=detection_reason,
                        port_scan_result=port_result,
                        dos_result=dos_result
                    )

                    print(
                        "[DASHBOARD] "
                        "Live prediction pushed."
                    )

                except Exception as error:

                    print(
                        "[DASHBOARD] "
                        f"Push failed: {error}"
                    )

            return {
                "ml_prediction":
                    ml_prediction,

                "ml_confidence":
                    ml_confidence,

                "behavioural_detection":
                    behavioural_detection,

                "final_decision":
                    final_decision,

                "detection_reason":
                    detection_reason,

                "port_scan":
                    port_result,

                "dos":
                    dos_result,

                "shap_explanation":
                    shap_explanation
            }

        except Exception as error:

            print(
                "[ML ERROR] "
                f"Live inference failed: {error}"
            )

            return None


if __name__ == "__main__":

    print(
        "This module defines the live "
        "inference engine."
    )

    print()

    print(
        "Run the project from the root "
        "directory."
    )

    print()

    print(
        "Example:"
    )

    print(
        "    python main.py capture"
    )