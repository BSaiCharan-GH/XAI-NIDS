from pathlib import Path
from collections import defaultdict, deque
import time

import joblib
import numpy as np
import pandas as pd


# ================================================================
# PORT SCAN DETECTOR
# ================================================================

class PortScanDetector:

    def __init__(
        self,
        window_seconds=5.0,
        minimum_unique_ports=10,
        cooldown_seconds=10.0,
    ):
        self.window_seconds = window_seconds
        self.minimum_unique_ports = minimum_unique_ports
        self.cooldown_seconds = cooldown_seconds

        self.connections = defaultdict(deque)
        self.last_alert = {}

    def process_flow(self, features):

        src_ip = features.get("src_ip")
        dst_ip = features.get("dst_ip")
        dst_port = features.get("dst_port")

        if not src_ip or not dst_ip or dst_port is None:
            return None

        try:
            dst_port = int(dst_port)
        except (ValueError, TypeError):
            return None

        capture_ts = features.get("capture_ts")

        if capture_ts is not None:

            try:
                now = pd.to_datetime(
                    capture_ts
                ).timestamp()

            except Exception:
                now = time.time()

        else:
            now = time.time()

        key = (
            str(src_ip),
            str(dst_ip)
        )

        self.connections[key].append(
            {
                "time": now,
                "port": dst_port
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

        flows = self.connections[key]

        if not flows:
            return None

        unique_ports = {
            item["port"]
            for item in flows
        }

        unique_port_count = len(
            unique_ports
        )

        connection_count = len(
            flows
        )

        time_span = max(
            now - flows[0]["time"],
            0.001
        )

        scan_rate = (
            connection_count /
            time_span
        )

        detected = (
            unique_port_count
            >=
            self.minimum_unique_ports
        )

        if not detected:

            return {
                "detected": False,
                "source": src_ip,
                "destination": dst_ip,
                "unique_ports": unique_port_count,
                "connections": connection_count,
                "scan_rate": scan_rate,
                "suppressed": False,
            }

        last_alert_time = self.last_alert.get(
            key,
            0
        )

        if (
            now - last_alert_time
            <
            self.cooldown_seconds
        ):

            return {
                "detected": True,
                "source": src_ip,
                "destination": dst_ip,
                "unique_ports": unique_port_count,
                "connections": connection_count,
                "scan_rate": scan_rate,
                "suppressed": True,
            }

        self.last_alert[key] = now

        return {
            "detected": True,
            "source": src_ip,
            "destination": dst_ip,
            "unique_ports": unique_port_count,
            "connections": connection_count,
            "scan_rate": scan_rate,
            "suppressed": False,
        }


# ================================================================
# SYN FLOOD / DOS DETECTOR
# ================================================================

class SYNFloodDetector:

    def __init__(
        self,
        window_seconds=5.0,
        minimum_syn_packets=20,
        minimum_syn_flows=10,
        minimum_syn_rate=10.0,
        maximum_ack_ratio=0.25,
        cooldown_seconds=10.0,
    ):

        self.window_seconds = window_seconds
        self.minimum_syn_packets = minimum_syn_packets
        self.minimum_syn_flows = minimum_syn_flows
        self.minimum_syn_rate = minimum_syn_rate
        self.maximum_ack_ratio = maximum_ack_ratio
        self.cooldown_seconds = cooldown_seconds

        self.connections = defaultdict(deque)
        self.last_alert = {}

    def process_flow(self, features):

        src_ip = features.get("src_ip")
        dst_ip = features.get("dst_ip")

        if not src_ip or not dst_ip:
            return None

        # --------------------------------------------------------
        # TCP only
        # --------------------------------------------------------

        try:

            protocol = int(
                features.get(
                    "protocol",
                    0
                )
            )

        except (
            ValueError,
            TypeError
        ):

            protocol = 0

        if protocol != 6:
            return None

        # --------------------------------------------------------
        # Safe numeric conversion
        # --------------------------------------------------------

        def number(
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

                if not np.isfinite(
                    value
                ):
                    return default

                return value

            except (
                ValueError,
                TypeError
            ):

                return default

        syn_count = number(
            "SYN Flag Count"
        )

        ack_count = number(
            "ACK Flag Count"
        )

        rst_count = number(
            "RST Flag Count"
        )

        fwd_packets = number(
            "Total Fwd Packets"
        )

        bwd_packets = number(
            "Total Backward Packets"
        )

        # --------------------------------------------------------
        # Only SYN-bearing flows participate
        # --------------------------------------------------------

        if syn_count <= 0:
            return None

        # --------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------

        capture_ts = features.get(
            "capture_ts"
        )

        if capture_ts is not None:

            try:

                now = pd.to_datetime(
                    capture_ts
                ).timestamp()

            except Exception:

                now = time.time()

        else:

            now = time.time()

        # --------------------------------------------------------
        # Source -> Destination
        # --------------------------------------------------------

        key = (
            str(src_ip),
            str(dst_ip)
        )

        self.connections[key].append(
            {
                "time": now,
                "syn": syn_count,
                "ack": ack_count,
                "rst": rst_count,
                "fwd": fwd_packets,
                "bwd": bwd_packets,
            }
        )

        # --------------------------------------------------------
        # Remove old observations
        # --------------------------------------------------------

        cutoff = (
            now -
            self.window_seconds
        )

        while (
            self.connections[key]
            and
            self.connections[key][0]["time"]
            <
            cutoff
        ):

            self.connections[key].popleft()

        observations = (
            self.connections[key]
        )

        if not observations:
            return None

        # --------------------------------------------------------
        # Aggregate statistics
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # SYN rate
        # --------------------------------------------------------

        elapsed = min(
            self.window_seconds,
            max(
                now -
                observations[0]["time"],
                0.1
            )
        )

        syn_rate = (
            total_syn /
            elapsed
        )

        # --------------------------------------------------------
        # ACK ratio
        # --------------------------------------------------------

        ack_ratio = (
            total_ack /
            max(
                total_syn +
                total_ack,
                1.0
            )
        )

        # --------------------------------------------------------
        # SYN ratio
        # --------------------------------------------------------

        syn_ratio = (
            total_syn /
            max(
                total_syn +
                total_ack,
                1.0
            )
        )

        # --------------------------------------------------------
        # One-way SYN flows
        # --------------------------------------------------------

        one_way_syn_flows = sum(
            1
            for item in observations
            if (
                item["syn"] > 0
                and
                item["bwd"] <= 0
            )
        )

        one_way_ratio = (
            one_way_syn_flows /
            max(
                syn_flows,
                1
            )
        )

        # --------------------------------------------------------
        # Detection conditions
        # --------------------------------------------------------

        enough_syn_packets = (
            total_syn
            >=
            self.minimum_syn_packets
        )

        enough_syn_flows = (
            syn_flows
            >=
            self.minimum_syn_flows
        )

        high_syn_rate = (
            syn_rate
            >=
            self.minimum_syn_rate
        )

        low_ack_ratio = (
            ack_ratio
            <=
            self.maximum_ack_ratio
        )

        detected = (
            enough_syn_packets
            and
            enough_syn_flows
            and
            high_syn_rate
            and
            low_ack_ratio
            and
            one_way_ratio >= 0.70
        )

        # --------------------------------------------------------
        # Not detected
        # --------------------------------------------------------

        if not detected:

            return {
                "detected": False,
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
                "suppressed": False,
            }

        # --------------------------------------------------------
        # Cooldown
        # --------------------------------------------------------

        last_alert = self.last_alert.get(
            key,
            0
        )

        if (
            now -
            last_alert
            <
            self.cooldown_seconds
        ):

            return {
                "detected": True,
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
                "suppressed": True,
            }

        # --------------------------------------------------------
        # New alert
        # --------------------------------------------------------

        self.last_alert[key] = now

        return {
            "detected": True,
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
            "suppressed": False,
        }


# ================================================================
# LIVE INFERENCE
# ================================================================

class LiveInference:

    def __init__(
        self,
        model_path="models/random_forest.pkl",
        dashboard_bridge=None,
    ):

        self.model_path = Path(
            model_path
        )

        self.dashboard_bridge = (
            dashboard_bridge
        )

        print()
        print("=" * 60)
        print(
            "XAI-IDS — Live Machine Learning Inference"
        )
        print("=" * 60)
        print()

        # --------------------------------------------------------
        # Load model
        # --------------------------------------------------------

        print(
            "[ML] Loading model..."
        )

        print(
            f"[ML] Model path: "
            f"{self.model_path.resolve()}"
        )

        model_data = joblib.load(
            self.model_path
        )

        self.model = model_data[
            "model"
        ]

        self.features = model_data[
            "features"
        ]

        print(
            f"[ML] Model type: "
            f"{type(self.model).__name__}"
        )

        print(
            f"[ML] Expected features: "
            f"{len(self.features)}"
        )

        # --------------------------------------------------------
        # Load SHAP
        # --------------------------------------------------------

        print(
            "[XAI] Loading SHAP..."
        )

        try:

            import shap

            self.shap = shap

            self.explainer = (
                shap.TreeExplainer(
                    self.model
                )
            )

            print(
                "[XAI] SHAP TreeExplainer "
                "initialized."
            )

        except Exception as exc:

            self.shap = None
            self.explainer = None

            print(
                "[XAI] SHAP initialization "
                f"failed: {exc}"
            )

        # --------------------------------------------------------
        # Port Scan detector
        # --------------------------------------------------------

        self.port_scan_detector = (
            PortScanDetector(
                window_seconds=5.0,
                minimum_unique_ports=10,
                cooldown_seconds=10.0,
            )
        )

        # --------------------------------------------------------
        # SYN Flood detector
        # --------------------------------------------------------

        self.syn_flood_detector = (
            SYNFloodDetector(
                window_seconds=5.0,
                minimum_syn_packets=20,
                minimum_syn_flows=10,
                minimum_syn_rate=10.0,
                maximum_ack_ratio=0.25,
                cooldown_seconds=10.0,
            )
        )

        print()
        print(
            "[SCAN] Port Scan detector initialized."
        )

        print(
            "[SCAN] Observation window : "
            "5 seconds"
        )

        print(
            "[SCAN] Minimum ports      : "
            "10"
        )

        print()
        print(
            "[DOS] SYN Flood detector initialized."
        )

        print(
            "[DOS] Observation window : "
            "5 seconds"
        )

        print(
            "[DOS] Minimum SYN packets: "
            "20"
        )

        print(
            "[DOS] Minimum SYN flows   : "
            "10"
        )

        print(
            "[DOS] Minimum SYN rate    : "
            "10/sec"
        )

        print(
            "[DOS] Maximum ACK ratio   : "
            "25%"
        )

        print()
        print(
            "[ML] Live inference ready."
        )

        print("=" * 60)
        print()


    # ============================================================
    # PROCESS FLOW
    # ============================================================

    def process_flow(
        self,
        features
    ):

        try:

            # ----------------------------------------------------
            # Build model input
            # ----------------------------------------------------

            row = {}

            for feature in self.features:

                value = features.get(
                    feature,
                    0
                )

                if value is None:
                    value = 0

                try:

                    value = float(
                        value
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    value = 0.0

                if not np.isfinite(
                    value
                ):

                    value = 0.0

                row[feature] = value

            X = pd.DataFrame(
                [row],
                columns=self.features
            )

            # ----------------------------------------------------
            # Random Forest prediction
            # ----------------------------------------------------

            prediction = (
                self.model.predict(X)[0]
            )

            prediction = str(
                prediction
            )

            # ----------------------------------------------------
            # Confidence
            # ----------------------------------------------------

            confidence_percent = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = (
                    self.model.predict_proba(
                        X
                    )[0]
                )

                confidence_percent = (
                    float(
                        np.max(
                            probabilities
                        )
                    )
                    *
                    100.0
                )

            # ----------------------------------------------------
            # SHAP
            # ----------------------------------------------------

            shap_explanation = (
                self._explain(
                    X,
                    prediction
                )
            )

            # ----------------------------------------------------
            # ML output
            # ----------------------------------------------------

            print()

            print(
                "  [ML] Prediction :",
                prediction
            )

            if confidence_percent is not None:

                print(
                    f"  [ML] Confidence : "
                    f"{confidence_percent:.2f}%"
                )

            # ====================================================
            # PORT SCAN DETECTION
            # ====================================================

            scan_result = (
                self.port_scan_detector
                .process_flow(
                    features
                )
            )

            # ====================================================
            # SYN FLOOD DETECTION
            # ====================================================

            syn_result = (
                self.syn_flood_detector
                .process_flow(
                    features
                )
            )

            # ====================================================
            # PORT SCAN ALERT
            # ====================================================

            port_scan_detected = (
                scan_result is not None
                and
                scan_result.get(
                    "detected",
                    False
                )
                and
                not scan_result.get(
                    "suppressed",
                    False
                )
            )

            if port_scan_detected:

                print()

                print(
                    "  [SCAN] "
                    "Behaviour analysis:"
                )

                print(
                    f"         Source       : "
                    f"{scan_result['source']}"
                )

                print(
                    f"         Destination  : "
                    f"{scan_result['destination']}"
                )

                print(
                    f"         Unique ports : "
                    f"{scan_result['unique_ports']}"
                )

                print(
                    f"         Connections  : "
                    f"{scan_result['connections']}"
                )

                print(
                    f"         Scan rate    : "
                    f"{scan_result['scan_rate']:.2f} "
                    f"flows/sec"
                )

                print()

                print(
                    "  [ALERT] "
                    "PORT SCAN DETECTED"
                )

            # ====================================================
            # DOS ALERT
            # ====================================================

            dos_detected = (
                syn_result is not None
                and
                syn_result.get(
                    "detected",
                    False
                )
                and
                not syn_result.get(
                    "suppressed",
                    False
                )
            )

            if dos_detected:

                print()

                print(
                    "  [DOS] "
                    "Behaviour analysis:"
                )

                print(
                    f"         Source       : "
                    f"{syn_result['source']}"
                )

                print(
                    f"         Destination  : "
                    f"{syn_result['destination']}"
                )

                print(
                    f"         SYN packets  : "
                    f"{syn_result['syn_packets']}"
                )

                print(
                    f"         Connections  : "
                    f"{syn_result['connections']}"
                )

                print(
                    f"         SYN rate     : "
                    f"{syn_result['syn_rate']:.2f} "
                    f"SYN/sec"
                )

                print(
                    f"         SYN ratio    : "
                    f"{syn_result['syn_ratio'] * 100:.2f}%"
                )

                print(
                    f"         RST packets  : "
                    f"{syn_result['rst_packets']}"
                )

                print(
                    f"         ACK packets  : "
                    f"{syn_result['ack_packets']}"
                )

                print()

                print(
                    "  [ALERT] "
                    "SYN FLOOD / DoS DETECTED"
                )

            # ====================================================
            # DECISION LAYERS
            # ====================================================

            ml_prediction = prediction

            if confidence_percent is not None:

                ml_confidence = (
                    confidence_percent /
                    100.0
                )

            else:

                ml_confidence = 0.0

            if dos_detected:

                behavioural_detection = (
                    "DoS"
                )

            elif port_scan_detected:

                behavioural_detection = (
                    "Port Scan"
                )

            else:

                behavioural_detection = (
                    "None"
                )

            # ----------------------------------------------------
            # Final security decision
            # ----------------------------------------------------

            if dos_detected:

                final_decision = (
                    "DoS"
                )

            elif port_scan_detected:

                final_decision = (
                    "Port Scan"
                )

            else:

                final_decision = (
                    ml_prediction
                )

            # ====================================================
            # PRINT DECISION
            # ====================================================

            print()

            print(
                "  [DECISION] "
                "======================================="
            )

            print(
                f"  [DECISION] ML Prediction          : "
                f"{ml_prediction}"
            )

            print(
                f"  [DECISION] ML Confidence          : "
                f"{ml_confidence * 100:.2f}%"
            )

            print(
                f"  [DECISION] Behavioural Detection  : "
                f"{behavioural_detection}"
            )

            print(
                f"  [DECISION] Final Security Decision: "
                f"{final_decision}"
            )

            print(
                "  [DECISION] "
                "======================================="
            )

            # ====================================================
            # DASHBOARD
            # ====================================================

            if self.dashboard_bridge is not None:

                try:

                    dashboard_flow = dict(
                        features
                    )

                    # ------------------------------------------------
                    # Decision layers
                    # ------------------------------------------------

                    dashboard_flow[
                        "ML Prediction"
                    ] = ml_prediction

                    dashboard_flow[
                        "ML Confidence"
                    ] = ml_confidence

                    dashboard_flow[
                        "Behavioural Detection"
                    ] = behavioural_detection

                    dashboard_flow[
                        "Final Security Decision"
                    ] = final_decision

                    # ------------------------------------------------
                    # Port Scan information
                    # ------------------------------------------------

                    if scan_result is not None:

                        dashboard_flow[
                            "Scan Detected"
                        ] = bool(
                            scan_result.get(
                                "detected",
                                False
                            )
                        )

                        dashboard_flow[
                            "Unique Scan Ports"
                        ] = int(
                            scan_result.get(
                                "unique_ports",
                                0
                            )
                        )

                        dashboard_flow[
                            "Scan Connections"
                        ] = int(
                            scan_result.get(
                                "connections",
                                0
                            )
                        )

                        dashboard_flow[
                            "Scan Rate"
                        ] = float(
                            scan_result.get(
                                "scan_rate",
                                0.0
                            )
                        )

                    else:

                        dashboard_flow[
                            "Scan Detected"
                        ] = False

                        dashboard_flow[
                            "Unique Scan Ports"
                        ] = 0

                        dashboard_flow[
                            "Scan Connections"
                        ] = 0

                        dashboard_flow[
                            "Scan Rate"
                        ] = 0.0

                    # ------------------------------------------------
                    # DoS information
                    # ------------------------------------------------

                    if syn_result is not None:

                        dashboard_flow[
                            "DoS Detected"
                        ] = bool(
                            syn_result.get(
                                "detected",
                                False
                            )
                        )

                        dashboard_flow[
                            "SYN Packets"
                        ] = int(
                            syn_result.get(
                                "syn_packets",
                                0
                            )
                        )

                        dashboard_flow[
                            "SYN Flows"
                        ] = int(
                            syn_result.get(
                                "syn_flows",
                                syn_result.get(
                                    "connections",
                                    0
                                )
                            )
                        )

                        dashboard_flow[
                            "SYN Rate"
                        ] = float(
                            syn_result.get(
                                "syn_rate",
                                0.0
                            )
                        )

                        dashboard_flow[
                            "SYN Ratio"
                        ] = float(
                            syn_result.get(
                                "syn_ratio",
                                0.0
                            )
                        )

                        dashboard_flow[
                            "ACK Ratio"
                        ] = float(
                            syn_result.get(
                                "ack_ratio",
                                0.0
                            )
                        )

                        dashboard_flow[
                            "RST Packets"
                        ] = int(
                            syn_result.get(
                                "rst_packets",
                                0
                            )
                        )

                        dashboard_flow[
                            "ACK Packets"
                        ] = int(
                            syn_result.get(
                                "ack_packets",
                                0
                            )
                        )

                        dashboard_flow[
                            "One Way Ratio"
                        ] = float(
                            syn_result.get(
                                "one_way_ratio",
                                0.0
                            )
                        )

                    else:

                        dashboard_flow[
                            "DoS Detected"
                        ] = False

                        dashboard_flow[
                            "SYN Packets"
                        ] = 0

                        dashboard_flow[
                            "SYN Flows"
                        ] = 0

                        dashboard_flow[
                            "SYN Rate"
                        ] = 0.0

                        dashboard_flow[
                            "SYN Ratio"
                        ] = 0.0

                        dashboard_flow[
                            "ACK Ratio"
                        ] = 0.0

                        dashboard_flow[
                            "RST Packets"
                        ] = 0

                        dashboard_flow[
                            "ACK Packets"
                        ] = 0

                        dashboard_flow[
                            "One Way Ratio"
                        ] = 0.0

                    # ------------------------------------------------
                    # Push result
                    # ------------------------------------------------

                    self.dashboard_bridge.push_prediction(
                        flow_data=dashboard_flow,
                        prediction=final_decision,
                        confidence=ml_confidence,
                        shap_explanation=shap_explanation,
                    )

                    print(
                        "  [DASHBOARD] "
                        "Live prediction pushed."
                    )

                except Exception as exc:

                    print(
                        f"  [DASHBOARD ERROR] "
                        f"{exc}"
                    )

        except Exception as exc:

            print(
                f"  [ML ERROR] "
                f"{exc}"
            )


    # ============================================================
    # SHAP
    # ============================================================

    def _explain(
        self,
        X,
        prediction
    ):

        if self.explainer is None:

            print(
                "  [XAI] SHAP unavailable."
            )

            return {}

        try:

            shap_values = (
                self.explainer.shap_values(
                    X
                )
            )

            values = (
                self._extract_shap_values(
                    shap_values,
                    prediction
                )
            )

            if values is None:

                print(
                    "  [XAI] "
                    "Could not extract SHAP values."
                )

                return {}

            values = np.asarray(
                values
            ).flatten()

            if len(values) != len(
                self.features
            ):

                print(
                    "  [XAI] "
                    "Feature/value mismatch."
                )

                return {}

            contributions = list(
                zip(
                    self.features,
                    values
                )
            )

            contributions.sort(
                key=lambda item:
                abs(item[1]),
                reverse=True
            )

            top_features = {
                feature: float(value)
                for feature, value
                in contributions[:10]
            }

            print()

            print(
                "  [XAI] "
                "Top contributing features:"
            )

            for feature, value in (
                contributions[:5]
            ):

                sign = (
                    "+"
                    if value >= 0
                    else "-"
                )

                print(
                    f"         "
                    f"{feature:<30} "
                    f"{sign}"
                    f"{abs(value):.4f}"
                )

            return top_features

        except Exception as exc:

            print(
                f"  [XAI ERROR] "
                f"{exc}"
            )

            return {}


    # ============================================================
    # SHAP FORMAT HANDLER
    # ============================================================

    def _extract_shap_values(
        self,
        shap_values,
        prediction
    ):

        if isinstance(
            shap_values,
            np.ndarray
        ):

            if shap_values.ndim == 3:

                try:

                    class_index = list(
                        self.model.classes_
                    ).index(
                        prediction
                    )

                except Exception:

                    class_index = 0

                return shap_values[
                    0,
                    :,
                    class_index
                ]

            if shap_values.ndim == 2:

                return shap_values[0]

            if shap_values.ndim == 1:

                return shap_values

        if isinstance(
            shap_values,
            list
        ):

            try:

                class_index = list(
                    self.model.classes_
                ).index(
                    prediction
                )

            except Exception:

                class_index = 0

            if class_index >= len(
                shap_values
            ):

                class_index = 0

            values = np.asarray(
                shap_values[
                    class_index
                ]
            )

            if values.ndim == 2:

                return values[0]

            if values.ndim == 1:

                return values

        return None


    # ============================================================
    # CALLABLE ALIAS
    # ============================================================

    def __call__(
        self,
        features
    ):

        return self.process_flow(
            features
        )