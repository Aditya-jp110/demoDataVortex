
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Big Data Analytics Platform",
    page_icon="🤖",
    layout="wide",
)

RANDOM_SEED = 42


# ============================================================
# Data Models
# ============================================================

@dataclass
class AgentResult:
    agent: str
    summary: str
    data: Dict[str, Any]


# ============================================================
# 1. Data Ingestion Simulation
# ============================================================

def generate_mock_data(days: int = 30, seed: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate:
      - structured SQL-like daily metrics
      - unstructured JSON-like system logs
    Includes an intentional revenue/user anomaly around day 22.
    """
    rng = np.random.default_rng(seed)
    start = datetime.now().date() - timedelta(days=days - 1)
    dates = pd.date_range(start=start, periods=days, freq="D")

    # Baseline business metrics
    trend = np.linspace(0, 8_500, days)
    weekly_seasonality = np.array([2_000, 1_300, 800, 1_600, 2_400, -1_200, -2_000])

    revenue = (
        95_000
        + trend
        + np.array([weekly_seasonality[i % 7] for i in range(days)])
        + rng.normal(0, 1_700, days)
    )

    active_users = (
        5_200
        + np.linspace(0, 450, days)
        + rng.normal(0, 110, days)
    )

    error_rate = np.clip(rng.normal(1.2, 0.25, days), 0.4, 3.0)
    avg_latency_ms = np.clip(rng.normal(180, 18, days), 110, 260)

    # Intentional anomaly: revenue and users drop sharply on day 22.
    anomaly_idx = 21
    revenue[anomaly_idx] *= 0.58
    active_users[anomaly_idx] *= 0.72
    error_rate[anomaly_idx] = 7.8
    avg_latency_ms[anomaly_idx] = 690

    sql_df = pd.DataFrame(
        {
            "date": dates,
            "revenue": np.round(revenue, 2),
            "active_users": np.round(active_users).astype(int),
            "error_rate_pct": np.round(error_rate, 2),
            "avg_latency_ms": np.round(avg_latency_ms, 1),
        }
    )

    # Unstructured logs stored as JSON-like records.
    logs: List[Dict[str, Any]] = []
    for i, dt in enumerate(dates):
        if i == anomaly_idx:
            messages = [
                "payment-service timeout spike; gateway requests failing",
                "db connection pool exhausted; retry queue growing",
                "checkout API latency critical; p95 above 1200ms",
                "5xx error burst detected on payment endpoint",
            ]
            severity = "CRITICAL"
        elif i in (20, 22):
            messages = [
                "payment gateway latency elevated",
                "database connection pool warning",
                "API retry count above normal baseline",
            ]
            severity = "WARN"
        else:
            messages = [
                "service healthy",
                "routine cache refresh completed",
                "scheduled analytics job completed",
            ]
            severity = "INFO"

        for msg in messages:
            logs.append(
                {
                    "timestamp": f"{dt.date()}T{rng.integers(0, 23):02d}:{rng.integers(0, 59):02d}:00",
                    "severity": severity,
                    "service": (
                        "payment-service"
                        if "payment" in msg or "checkout" in msg
                        else "database"
                        if "db" in msg or "database" in msg
                        else "api"
                    ),
                    "message": msg,
                    "metadata": json.dumps(
                        {
                            "region": "central-india",
                            "instance": f"node-{rng.integers(1, 5)}",
                            "trace_id": f"trace-{rng.integers(100000, 999999)}",
                        }
                    ),
                }
            )

    logs_df = pd.DataFrame(logs)
    logs_df["date"] = pd.to_datetime(logs_df["timestamp"]).dt.normalize()

    return sql_df, logs_df


# ============================================================
# 2. Simulated Distributed Processing Layer (PySpark-like)
# ============================================================

def simulate_pyspark_processing(sql_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mimics a distributed processing job:
      - schema/duplicate/null checks
      - type normalization
      - rolling statistics
      - z-score anomaly tagging

    This intentionally uses pandas so the prototype runs without
    installing a real Spark cluster.
    """
    df = sql_df.copy()

    # --- Cleaning ---
    numeric_cols = [
        "revenue",
        "active_users",
        "error_rate_pct",
        "avg_latency_ms",
    ]
    df["date"] = pd.to_datetime(df["date"])

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    df[numeric_cols] = df[numeric_cols].interpolate().ffill().bfill()

    # --- Rolling averages ---
    df["revenue_7d_avg"] = df["revenue"].rolling(7, min_periods=3).mean()
    df["users_7d_avg"] = df["active_users"].rolling(7, min_periods=3).mean()

    # --- Anomaly score ---
    # Use rolling baseline to simulate a distributed feature-engineering step.
    baseline = df["revenue"].rolling(7, min_periods=5).mean()
    rolling_std = df["revenue"].rolling(7, min_periods=5).std().replace(0, np.nan)

    df["revenue_zscore"] = ((df["revenue"] - baseline) / rolling_std).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    df["anomaly"] = df["revenue_zscore"].abs() >= 2.5

    return df


# ============================================================
# 3. Multi-Agent Swarm
# ============================================================

class DataQueryAgent:
    name = "Data Query Agent"

    def run(self, processed_df: pd.DataFrame) -> AgentResult:
        revenue = processed_df["revenue"]
        users = processed_df["active_users"]

        highest_row = processed_df.loc[revenue.idxmax()]
        lowest_row = processed_df.loc[revenue.idxmin()]

        metrics = {
            "revenue_total": float(revenue.sum()),
            "revenue_avg": float(revenue.mean()),
            "revenue_min": float(revenue.min()),
            "revenue_max": float(revenue.max()),
            "users_avg": float(users.mean()),
            "users_min": int(users.min()),
            "users_max": int(users.max()),
            "highest_revenue_date": highest_row["date"].strftime("%Y-%m-%d"),
            "lowest_revenue_date": lowest_row["date"].strftime("%Y-%m-%d"),
            "anomaly_count": int(processed_df["anomaly"].sum()),
        }

        summary = (
            f"30-day revenue totals ₹{metrics['revenue_total']:,.0f}; "
            f"average ₹{metrics['revenue_avg']:,.0f}/day. "
            f"Revenue ranged from ₹{metrics['revenue_min']:,.0f} to "
            f"₹{metrics['revenue_max']:,.0f}. "
            f"Average active users: {metrics['users_avg']:,.0f}."
        )

        return AgentResult(self.name, summary, metrics)


class PatternTrendDiscoveryAgent:
    name = "Pattern & Trend Discovery Agent"

    def run(self, processed_df: pd.DataFrame, logs_df: pd.DataFrame) -> AgentResult:
        anomalies = processed_df[processed_df["anomaly"]].copy()

        mappings: List[Dict[str, Any]] = []
        for _, row in anomalies.iterrows():
            d = row["date"].normalize()
            day_logs = logs_df[logs_df["date"] == d]

            # Score likely root causes by keyword hits.
            text_blob = " ".join(day_logs["message"].astype(str)).lower()
            candidates = {
                "Payment gateway / checkout failure": sum(
                    k in text_blob
                    for k in ["payment", "gateway", "checkout", "5xx", "timeout"]
                ),
                "Database connection saturation": sum(
                    k in text_blob for k in ["db", "database", "connection pool", "retry queue"]
                ),
                "API latency / service degradation": sum(
                    k in text_blob for k in ["latency", "api", "error", "timeout"]
                ),
            }

            root_cause = max(candidates, key=candidates.get)
            mappings.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "revenue": float(row["revenue"]),
                    "users": int(row["active_users"]),
                    "zscore": float(row["revenue_zscore"]),
                    "root_cause": root_cause,
                    "critical_logs": int((day_logs["severity"] == "CRITICAL").sum()),
                }
            )

        # Fallback when no anomaly is detected.
        if not mappings:
            conclusion = "No statistically significant revenue anomaly was detected."
        else:
            top = max(mappings, key=lambda x: abs(x["zscore"]))
            conclusion = (
                f"Detected {len(mappings)} anomalous day(s). The strongest anomaly occurred on "
                f"{top['date']} and correlates with {top['root_cause'].lower()}."
            )

        return AgentResult(
            self.name,
            conclusion,
            {"anomalies": mappings},
        )


class PredictiveMLAgent:
    name = "Predictive ML Agent"

    def run(self, processed_df: pd.DataFrame, horizon: int = 3) -> AgentResult:
        horizon = int(np.clip(horizon, 3, 7))
        df = processed_df.copy()

        # Lightweight linear trend model:
        # y = a*x + b
        x = np.arange(len(df), dtype=float)
        y = df["revenue"].to_numpy(dtype=float)

        slope, intercept = np.polyfit(x, y, 1)
        future_x = np.arange(len(df), len(df) + horizon, dtype=float)
        forecast = slope * future_x + intercept

        last_date = pd.to_datetime(df["date"].iloc[-1])
        future_dates = [
            last_date + timedelta(days=i) for i in range(1, horizon + 1)
        ]

        forecast_df = pd.DataFrame(
            {
                "date": future_dates,
                "revenue_forecast": np.maximum(forecast, 0),
            }
        )

        direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"
        summary = (
            f"Using a lightweight linear trend model, the next {horizon} days show a "
            f"{direction} revenue trajectory with an average forecast of "
            f"₹{forecast_df['revenue_forecast'].mean():,.0f}/day."
        )

        return AgentResult(
            self.name,
            summary,
            {
                "horizon": horizon,
                "slope": float(slope),
                "forecast_df": forecast_df,
            },
        )


class SynthesizerReportingAgent:
    name = "Synthesizer / Reporting Agent"

    def run(
        self,
        query: str,
        query_result: AgentResult,
        pattern_result: AgentResult,
        ml_result: AgentResult,
    ) -> AgentResult:

        anomalies = pattern_result.data.get("anomalies", [])
        forecast_df = ml_result.data["forecast_df"]

        anomaly_text = "No significant anomalies were found."
        root_cause_text = "No root-cause correlation was necessary."

        if anomalies:
            strongest = max(anomalies, key=lambda x: abs(x["zscore"]))
            anomaly_text = (
                f"Anomaly detected on **{strongest['date']}**: revenue was "
                f"₹{strongest['revenue']:,.0f} with a revenue z-score of "
                f"{strongest['zscore']:.2f}, and active users fell to "
                f"{strongest['users']:,}."
            )
            root_cause_text = (
                f"Correlated logs point to **{strongest['root_cause']}**, with "
                f"{strongest['critical_logs']} critical log event(s) that day."
            )

        forecast_avg = float(forecast_df["revenue_forecast"].mean())
        next_day = float(forecast_df["revenue_forecast"].iloc[0])
        last_actual = float(query_result.data["revenue_avg"])

        recommendations = [
            "Review payment gateway and checkout telemetry around the anomaly window.",
            "Add alerting on payment 5xx rate, connection-pool saturation, and API latency.",
            "Compare forecast against actuals daily and retrain/replace the simple trend model when more history is available.",
        ]

        report = f"""
## Executive Analytics Report

**User query:** {query}

### 1. Business snapshot
- 30-day revenue: **₹{query_result.data['revenue_total']:,.0f}**
- Average daily revenue: **₹{query_result.data['revenue_avg']:,.0f}**
- Average active users: **{query_result.data['users_avg']:,.0f}**
- Detected anomaly days: **{query_result.data['anomaly_count']}**

### 2. Anomaly & root-cause analysis
{anomaly_text}

{root_cause_text}

### 3. Forecast
The next **{ml_result.data['horizon']} days** have an average forecast of
**₹{forecast_avg:,.0f}/day**, versus a 30-day historical average of
**₹{last_actual:,.0f}/day**.

Forecast begins at approximately **₹{next_day:,.0f}** on
**{forecast_df['date'].iloc[0].strftime('%Y-%m-%d')}**.

### 4. Recommended actions
1. {recommendations[0]}
2. {recommendations[1]}
3. {recommendations[2]}

> **Prototype note:** This is a local simulation. The processing layer mimics PySpark,
> and the prediction layer intentionally uses a lightweight trend model so the entire
> platform runs without a Spark cluster, database, or cloud credentials.
""".strip()

        return AgentResult(
            self.name,
            "Synthesized business metrics, root-cause findings, and forecast into an executive report.",
            {"report": report},
        )


# ============================================================
# 4. Natural-Language Query Router
# ============================================================

def parse_user_query(query: str) -> Dict[str, Any]:
    """
    Lightweight intent extraction:
      - forecast horizon 3..7 days
      - request type: broad / anomaly / forecast / metrics
    """
    q = query.lower().strip()

    match = re.search(r"next\s+([3-7])\s+days?", q)
    horizon = int(match.group(1)) if match else 3

    if any(k in q for k in ["cause", "root cause", "why", "anomal", "drop"]):
        intent = "anomaly"
    elif any(k in q for k in ["predict", "forecast", "future", "next"]):
        intent = "forecast"
    elif any(k in q for k in ["sum", "average", "avg", "highest", "lowest", "metric"]):
        intent = "metrics"
    else:
        intent = "broad"

    return {"intent": intent, "forecast_horizon": horizon}


# ============================================================
# 5. Agent Orchestrator
# ============================================================

class AnalyticsSwarm:
    def __init__(self):
        self.query_agent = DataQueryAgent()
        self.pattern_agent = PatternTrendDiscoveryAgent()
        self.ml_agent = PredictiveMLAgent()
        self.synth_agent = SynthesizerReportingAgent()

    def execute(
        self,
        query: str,
        processed_df: pd.DataFrame,
        logs_df: pd.DataFrame,
        horizon: int,
        status_box,
    ) -> Dict[str, AgentResult]:

        results: Dict[str, AgentResult] = {}

        steps = [
            ("Data Query Agent", lambda: self.query_agent.run(processed_df)),
            ("Pattern & Trend Discovery Agent", lambda: self.pattern_agent.run(processed_df, logs_df)),
            ("Predictive ML Agent", lambda: self.ml_agent.run(processed_df, horizon)),
        ]

        progress = status_box.progress(0)
        for idx, (name, fn) in enumerate(steps, start=1):
            status_box.write(f"🔄 Running **{name}**...")
            time.sleep(0.35)
            result = fn()
            results[result.agent] = result
            progress.progress(int(idx / (len(steps) + 1) * 100))
            status_box.write(f"✅ {name} completed")

        status_box.write("🔄 Running **Synthesizer / Reporting Agent**...")
        time.sleep(0.35)
        synth = self.synth_agent.run(
            query,
            results["Data Query Agent"],
            results["Pattern & Trend Discovery Agent"],
            results["Predictive ML Agent"],
        )
        results[synth.agent] = synth
        progress.progress(100)
        status_box.write("✅ Synthesizer / Reporting Agent completed")

        return results


# ============================================================
# 6. Visualization
# ============================================================

def build_forecast_chart(processed_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=processed_df["date"],
            y=processed_df["revenue"],
            mode="lines+markers",
            name="Historical Revenue",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=processed_df["date"],
            y=processed_df["revenue_7d_avg"],
            mode="lines",
            name="7-Day Rolling Average",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["revenue_forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        title="Revenue History + Rolling Average + Forecast",
        xaxis_title="Date",
        yaxis_title="Revenue (₹)",
        hovermode="x unified",
        height=500,
    )

    return fig


# ============================================================
# 7. Streamlit UI
# ============================================================

def main():
    st.title("🤖 Multi-Agent Big Data Analytics Platform")
    st.caption(
        "Local end-to-end prototype: ingestion → processing → agent swarm → forecast → executive report"
    )

    # Session state
    if "sql_df" not in st.session_state:
        st.session_state.sql_df = None
    if "logs_df" not in st.session_state:
        st.session_state.logs_df = None
    if "processed_df" not in st.session_state:
        st.session_state.processed_df = None
    if "results" not in st.session_state:
        st.session_state.results = None

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        generate = st.button("Generate / Refresh Mock Data", use_container_width=True)
        horizon = st.slider(
            "Forecast horizon (days)",
            min_value=3,
            max_value=7,
            value=3,
        )

        st.divider()
        st.markdown(
            """
            **Prototype stack**

            - Python + Streamlit
            - Pandas / NumPy
            - Plotly
            - Modular agent classes
            - JSON-like logs
            - Simulated PySpark processing
            - Local trend forecasting
            """
        )

    if generate or st.session_state.sql_df is None:
        sql_df, logs_df = generate_mock_data()
        processed_df = simulate_pyspark_processing(sql_df)

        st.session_state.sql_df = sql_df
        st.session_state.logs_df = logs_df
        st.session_state.processed_df = processed_df

    sql_df = st.session_state.sql_df
    logs_df = st.session_state.logs_df
    processed_df = st.session_state.processed_df

    # Top KPIs
    anomaly_count = int(processed_df["anomaly"].sum())
    total_revenue = float(processed_df["revenue"].sum())
    avg_users = float(processed_df["active_users"].mean())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("30-Day Revenue", f"₹{total_revenue:,.0f}")
    k2.metric("Avg Active Users", f"{avg_users:,.0f}")
    k3.metric("Anomaly Days", str(anomaly_count))
    k4.metric("Critical Log Events", str(int((logs_df["severity"] == "CRITICAL").sum())))

    # Data tabs
    tab1, tab2, tab3 = st.tabs(["📊 Structured Metrics", "🧾 Unstructured Logs", "⚙️ Processed Features"])

    with tab1:
        st.dataframe(sql_df, use_container_width=True, hide_index=True)

    with tab2:
        st.dataframe(
            logs_df[["timestamp", "severity", "service", "message", "metadata"]],
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        st.dataframe(processed_df, use_container_width=True, hide_index=True)

    # Natural-language query
    st.subheader("💬 Ask the Analytics Swarm")
    default_query = "Find the revenue drop cause and predict the next 3 days"
    query = st.text_input(
        "Natural-language question",
        value=default_query,
        placeholder="Example: Find the revenue drop cause and predict the next 5 days",
    )

    run_query = st.button("🚀 Run Multi-Agent Analysis", type="primary")

    if run_query:
        parsed = parse_user_query(query)
        chosen_horizon = parsed["forecast_horizon"] if "next" in query.lower() else horizon

        status = st.status("Executing multi-agent workflow...", expanded=True)

        swarm = AnalyticsSwarm()
        results = swarm.execute(
            query=query,
            processed_df=processed_df,
            logs_df=logs_df,
            horizon=chosen_horizon,
            status_box=status,
        )

        st.session_state.results = results
        status.update(label="Multi-agent workflow completed", state="complete")

    # Display results
    results = st.session_state.results

    if results:
        st.subheader("🧠 Agent Execution Results")

        agent_cols = st.columns(4)
        agent_order = [
            "Data Query Agent",
            "Pattern & Trend Discovery Agent",
            "Predictive ML Agent",
            "Synthesizer / Reporting Agent",
        ]

        for col, agent_name in zip(agent_cols, agent_order):
            with col:
                result = results[agent_name]
                st.markdown(f"### {agent_name}")
                st.write(result.summary)

        st.subheader("📈 Revenue Forecast")
        forecast_df = results["Predictive ML Agent"].data["forecast_df"]
        st.plotly_chart(
            build_forecast_chart(processed_df, forecast_df),
            use_container_width=True,
        )

        st.subheader("🚨 Detected Anomalies")
        anomalies = results["Pattern & Trend Discovery Agent"].data["anomalies"]
        if anomalies:
            st.dataframe(pd.DataFrame(anomalies), use_container_width=True, hide_index=True)
        else:
            st.info("No anomalies detected.")

        st.subheader("📝 Executive Report")
        st.markdown(results["Synthesizer / Reporting Agent"].data["report"])

        # Show query routing for transparency
        with st.expander("🔍 Query Router Details"):
            st.json(parse_user_query(query))

    st.divider()
    st.caption(
        f"Prototype generated locally • Seed={RANDOM_SEED} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
