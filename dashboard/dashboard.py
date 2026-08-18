import sqlite3
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard_bridge import bridge_instance

st.set_page_config(page_title="XAI-NIDS Security Dashboard", page_icon="🛡️", layout="wide")

st.markdown("# Explainable AI Network Intrusion Detection System")
st.caption("Real-time network monitoring, behavioural threat detection, Random Forest inference and SHAP explanations")


def protocol_name(value):
    try:
        return {1: "ICMP", 6: "TCP", 17: "UDP"}.get(int(float(value)), str(value))
    except (TypeError, ValueError):
        return str(value)


def confidence_text(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def decision_counts(flows):
    counts = {"Benign": 0, "DoS": 0, "Port Scan": 0}
    for flow in flows:
        label = str(flow.get("final_decision", flow.get("prediction", "Unknown")))
        if label in counts:
            counts[label] += 1
    return counts


st.sidebar.header("Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 1, 5, 1)

snapshot = bridge_instance.get_snapshot()
recent_flows = snapshot.get("recent_flows", [])
alerts = snapshot.get("alerts", [])
counts = decision_counts(recent_flows)

st.markdown("### System Overview")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Packets Processed", snapshot.get("total_packets", 0))
m2.metric("Active Flows", snapshot.get("active_flows", 0))
m3.metric("Threat Alerts", len(alerts))
m4.metric("Port Scans", counts["Port Scan"])
m5.metric("DoS Attacks", counts["DoS"])
st.caption(f"Benign flows: {counts['Benign']} | Recent flows tracked: {len(recent_flows)}")
st.markdown("---")

if alerts:
    latest = alerts[0]
    st.error(
        f"LATEST SECURITY EVENT — {latest.get('final_decision', latest.get('prediction', 'Unknown'))} | "
        f"Reason: {latest.get('detection_reason', 'N/A')} | "
        f"{latest.get('src_ip')}:{latest.get('src_port')} → {latest.get('dst_ip')}:{latest.get('dst_port')}"
    )
else:
    st.success("SYSTEM STATUS — No malicious traffic detected in the current live session.")

tab_live, tab_xai, tab_history = st.tabs(["Live Traffic & Alerts", "XAI Explanations", "Historical Logs"])

with tab_live:
    st.subheader("Live Analyzed Flows")
    if recent_flows:
        df = pd.DataFrame(recent_flows)
        display_cols = [
            "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
            "ml_prediction", "confidence", "behavioural_detection", "final_decision", "detection_reason"
        ]
        available = [c for c in display_cols if c in df.columns]
        table = df[available].copy()
        if "protocol" in table:
            table["protocol"] = table["protocol"].apply(protocol_name)
        if "confidence" in table:
            table["confidence"] = table["confidence"].apply(confidence_text)
        st.dataframe(table, use_container_width=True, height=430, hide_index=True)
    else:
        st.info("No network flows analyzed yet. Waiting for live traffic...")

    st.markdown("### Recent Threat Feed")
    if alerts:
        for alert in alerts[:8]:
            label = alert.get("final_decision", alert.get("prediction", "Unknown"))
            with st.expander(f"{label} | {alert.get('timestamp', 'N/A')}", expanded=(alert is alerts[0])):
                c1, c2, c3, c4 = st.columns(4)
                c1.write(f"**Source**\n{alert.get('src_ip')}:{alert.get('src_port')}")
                c2.write(f"**Destination**\n{alert.get('dst_ip')}:{alert.get('dst_port')}")
                c3.write(f"**ML Prediction**\n{alert.get('ml_prediction', 'N/A')} ({confidence_text(alert.get('confidence'))})")
                c4.write(f"**Behaviour / Final**\n{alert.get('behavioural_detection', 'N/A')} → {label}")
                st.write(f"**Reason:** {alert.get('detection_reason', 'N/A')}")
    else:
        st.success("No active threat alerts detected.")

with tab_xai:
    st.subheader("Why did the model make this prediction?")
    if recent_flows:
        flow_options = [
            f"[{f.get('timestamp', 'N/A')}] {f.get('src_ip')}:{f.get('src_port')} → {f.get('dst_ip')}:{f.get('dst_port')} | {f.get('final_decision', f.get('prediction', 'UNKNOWN'))}"
            for f in recent_flows
        ]
        selected_idx = st.selectbox("Select a captured flow:", range(len(flow_options)), format_func=lambda i: flow_options[i])
        selected = recent_flows[selected_idx]
        ml_prediction = selected.get("ml_prediction", selected.get("prediction", "Unknown"))
        final_decision = selected.get("final_decision", selected.get("prediction", "Unknown"))
        behavioural = selected.get("behavioural_detection", "Benign")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ML Prediction", ml_prediction)
        c2.metric("Confidence", confidence_text(selected.get("confidence")))
        c3.metric("Behavioural Detection", behavioural)
        c4.metric("Final Security Decision", final_decision)
        st.info(f"Decision reason: {selected.get('detection_reason', 'N/A')}")

        shap_dict = selected.get("shap_explanation", {})
        if shap_dict:
            sorted_shap = sorted(shap_dict.items(), key=lambda item: abs(float(item[1])), reverse=True)[:10]
            features = [x[0] for x in sorted_shap]
            values = [float(x[1]) for x in sorted_shap]
            fig = go.Figure(go.Bar(x=values, y=features, orientation="h", hovertemplate="%{y}<br>SHAP value: %{x:.4f}<extra></extra>"))
            fig.update_layout(title=f"Top 10 SHAP Contributions — ML class: {ml_prediction}", xaxis_title="SHAP contribution", yaxis_title="Network feature", yaxis=dict(autorange="reversed"), height=480)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("SHAP values explain the Random Forest prediction. Behavioural detectors are rule-based and are not inferred from SHAP.")
            for rank, (feature, value) in enumerate(sorted_shap[:5], start=1):
                direction = "toward" if float(value) >= 0 else "away from"
                st.write(f"**{rank}. {feature}** — {float(value):+.4f}, pushing the model {direction} {ml_prediction}.")
        else:
            st.info("SHAP explanation is not available for this flow.")
    else:
        st.info("No flow data available for explanation.")

with tab_history:
    st.subheader("Persisted Security Events")
    try:
        with sqlite3.connect(bridge_instance._db_path) as conn:
            df_hist = pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC LIMIT 500", conn)
        if not df_hist.empty:
            labels = ["All"] + sorted(df_hist["final_decision"].dropna().unique().tolist())
            selected_label = st.selectbox("Filter by final decision", labels)
            search_ip = st.text_input("Filter by source or destination IP")
            filtered = df_hist.copy()
            if selected_label != "All":
                filtered = filtered[filtered["final_decision"] == selected_label]
            if search_ip.strip():
                term = search_ip.strip()
                filtered = filtered[
                    filtered["src_ip"].astype(str).str.contains(term, case=False, na=False)
                    | filtered["dst_ip"].astype(str).str.contains(term, case=False, na=False)
                ]
            st.dataframe(filtered, use_container_width=True, hide_index=True)
            st.download_button("Download Filtered Security Log", filtered.to_csv(index=False).encode("utf-8"), "xai_ids_threat_log.csv", "text/csv")
        else:
            st.info("No malicious events have been stored yet.")
    except Exception as exc:
        st.error(f"Failed to query database: {exc}")

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
