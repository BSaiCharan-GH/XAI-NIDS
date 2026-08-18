import sqlite3
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard_bridge import bridge_instance

st.set_page_config(
    page_title="XAI-NIDS Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.2rem; }
    .subtitle { color: #6b7280; margin-bottom: 1rem; }
    .threat-box { padding: 0.8rem 1rem; border-radius: 0.6rem; border: 1px solid #e5e7eb; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Explainable AI Network Intrusion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time network monitoring, threat detection and model explainability</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def protocol_name(value):
    try:
        value = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    return {1: "ICMP", 6: "TCP", 17: "UDP"}.get(value, str(value))


def confidence_text(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def threat_counts(flows):
    counts = {"Benign": 0, "DoS": 0, "Port Scan": 0}
    for flow in flows:
        label = str(flow.get("prediction", "Unknown"))
        if label in counts:
            counts[label] += 1
    return counts


def explain_direction(value, prediction):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "No measurable contribution"

    if value >= 0:
        return f"pushes the model toward {prediction}"
    return f"pushes the model away from {prediction}"


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 1, 5, 1)

snapshot = bridge_instance.get_snapshot()
recent_flows = snapshot.get("recent_flows", [])
alerts = snapshot.get("alerts", [])
counts = threat_counts(recent_flows)

# -----------------------------------------------------------------------------
# System overview
# -----------------------------------------------------------------------------
st.markdown("### System Overview")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Packets Processed", snapshot.get("total_packets", 0))
with m2:
    st.metric("Active Flows", snapshot.get("active_flows", 0))
with m3:
    st.metric("Threat Alerts", len(alerts))
with m4:
    st.metric("Port Scans", counts["Port Scan"])
with m5:
    st.metric("DoS Attacks", counts["DoS"])

st.caption("Benign flows: {}  |  Recent flows tracked: {}".format(counts["Benign"], len(recent_flows)))
st.markdown("---")

# -----------------------------------------------------------------------------
# Latest event banner
# -----------------------------------------------------------------------------
if alerts:
    latest = alerts[0]
    st.error(
        f"LATEST SECURITY EVENT — {latest.get('prediction', 'Unknown')} | "
        f"{latest.get('src_ip')}:{latest.get('src_port')} → "
        f"{latest.get('dst_ip')}:{latest.get('dst_port')} | "
        f"Confidence: {confidence_text(latest.get('confidence'))}"
    )
else:
    st.success("SYSTEM STATUS — No malicious traffic has been detected in the current live session.")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_live, tab_xai, tab_history = st.tabs(
    ["Live Traffic & Alerts", "XAI Explanations", "Historical Logs"]
)

# =============================================================================
# TAB 1 — LIVE TRAFFIC
# =============================================================================
with tab_live:
    st.subheader("Live Analyzed Flows")

    if recent_flows:
        df = pd.DataFrame(recent_flows)

        display_cols = [
            "timestamp",
            "src_ip",
            "src_port",
            "dst_ip",
            "dst_port",
            "protocol",
            "prediction",
            "confidence",
        ]

        available = [c for c in display_cols if c in df.columns]
        table = df[available].copy()

        if "protocol" in table.columns:
            table["protocol"] = table["protocol"].apply(protocol_name)
        if "confidence" in table.columns:
            table["confidence"] = table["confidence"].apply(confidence_text)

        st.dataframe(
            table,
            use_container_width=True,
            height=430,
            hide_index=True,
        )
    else:
        st.info("No network flows analyzed yet. Waiting for live traffic...")

    st.markdown("### Recent Threat Feed")

    if alerts:
        for alert in alerts[:8]:
            label = str(alert.get("prediction", "Unknown"))
            with st.expander(
                f"{label}  |  {confidence_text(alert.get('confidence'))}  |  {alert.get('timestamp', 'N/A')}",
                expanded=(alert is alerts[0]),
            ):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Source**  \n{alert.get('src_ip')}:{alert.get('src_port')}")
                c2.write(f"**Destination**  \n{alert.get('dst_ip')}:{alert.get('dst_port')}")
                c3.write(f"**Protocol**  \n{protocol_name(alert.get('protocol'))}")
    else:
        st.success("No active threat alerts detected.")

# =============================================================================
# TAB 2 — XAI
# =============================================================================
with tab_xai:
    st.subheader("Why did the model make this prediction?")
    st.write(
        "Select a live flow to inspect its Random Forest prediction and the "
        "SHAP feature contributions for that specific flow."
    )

    if recent_flows:
        flow_options = []
        for i, flow in enumerate(recent_flows):
            flow_options.append(
                f"[{flow.get('timestamp', 'N/A')}] "
                f"{flow.get('src_ip')}:{flow.get('src_port')} → "
                f"{flow.get('dst_ip')}:{flow.get('dst_port')} | "
                f"{flow.get('prediction', 'UNKNOWN')}"
            )

        selected_idx = st.selectbox(
            "Select a captured flow:",
            options=range(len(flow_options)),
            format_func=lambda i: flow_options[i],
            key="flow_selector",
        )

        selected = recent_flows[selected_idx]
        prediction = str(selected.get("prediction", "Unknown"))

        st.markdown("### Decision Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Label", prediction)
        c2.metric("Confidence", confidence_text(selected.get("confidence")))
        c3.metric("Protocol", protocol_name(selected.get("protocol")))
        c4.metric("Destination Port", selected.get("dst_port", "N/A"))

        shap_dict = selected.get("shap_explanation", {})

        st.markdown("### Feature Contributions")

        if shap_dict:
            sorted_shap = sorted(
                shap_dict.items(),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )[:10]

            features = [item[0] for item in sorted_shap]
            values = [float(item[1]) for item in sorted_shap]

            fig = go.Figure(
                go.Bar(
                    x=values,
                    y=features,
                    orientation="h",
                    marker_color=[
                        "#d62728" if value >= 0 else "#1f77b4"
                        for value in values
                    ],
                    hovertemplate="%{y}<br>SHAP value: %{x:.4f}<extra></extra>",
                )
            )
            fig.update_layout(
                title=f"Top 10 SHAP Contributions — {prediction}",
                xaxis_title="SHAP contribution",
                yaxis_title="Network feature",
                yaxis=dict(autorange="reversed"),
                height=480,
                margin=dict(l=10, r=20, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Positive values push the prediction toward the selected class; "
                "negative values push it away from the selected class."
            )

            st.markdown("### Explanation Summary")
            top = sorted_shap[:5]
            for rank, (feature, value) in enumerate(top, start=1):
                st.write(
                    f"**{rank}. {feature}** — {float(value):+.4f}; "
                    f"this {explain_direction(value, prediction)}."
                )
        else:
            st.info("SHAP explanation is not available for this flow record.")
    else:
        st.info("No flow data available for explanation.")

# =============================================================================
# TAB 3 — HISTORICAL LOGS
# =============================================================================
with tab_history:
    st.subheader("Persisted Security Events")
    st.caption("Historical malicious detections stored in SQLite.")

    try:
        with sqlite3.connect(bridge_instance._db_path) as conn:
            df_hist = pd.read_sql_query(
                "SELECT * FROM detections ORDER BY id DESC LIMIT 500",
                conn,
            )

        if not df_hist.empty:
            f1, f2 = st.columns(2)
            with f1:
                labels = ["All"] + sorted(df_hist["prediction"].dropna().unique().tolist())
                selected_label = st.selectbox("Filter by detection", labels)
            with f2:
                search_ip = st.text_input("Filter by source or destination IP")

            filtered = df_hist.copy()

            if selected_label != "All":
                filtered = filtered[filtered["prediction"] == selected_label]

            if search_ip.strip():
                mask = (
                    filtered["src_ip"].astype(str).str.contains(search_ip.strip(), case=False, na=False)
                    | filtered["dst_ip"].astype(str).str.contains(search_ip.strip(), case=False, na=False)
                )
                filtered = filtered[mask]

            st.dataframe(filtered, use_container_width=True, hide_index=True)

            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Filtered Security Log",
                data=csv,
                file_name="xai_ids_threat_log.csv",
                mime="text/csv",
            )
        else:
            st.info("No malicious events have been stored yet.")

    except Exception as exc:
        st.error(f"Failed to query database: {exc}")

# -----------------------------------------------------------------------------
# Auto refresh
# -----------------------------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
