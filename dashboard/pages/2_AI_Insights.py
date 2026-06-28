"""
AI Insights Page (v2)
- Works with structured AgentResponse from safe_process()
- Shows summary + insights + recommendations upfront
- Full LLM response in expandable section
- Handles both old dict responses and new AgentResponse objects
"""
import streamlit as st
import pandas as pd
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

from database.db_manager import db_manager
from agents.orchestrator import AgentOrchestrator
from agents.response_model import AgentResponse, AgentStatus, InsightSeverity

st.set_page_config(page_title="AI Insights", page_icon="🤖", layout="wide")

st.title("🤖 AI-Powered Insights")
st.markdown("Get intelligent recommendations from our multi-agent AI system")


# ------------------------------------------------------------------
# Initialize orchestrator
# ------------------------------------------------------------------
@st.cache_resource
def get_orchestrator():
    return AgentOrchestrator()

orchestrator = get_orchestrator()


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("AI Agent Selection")
agent_type = st.sidebar.radio(
    "Choose an agent:",
    ["📊 Demand Forecasting", "📦 Inventory Optimization",
     "⚠️ Anomaly Detection", "🔄 All Agents (Comprehensive)"]
)

st.sidebar.markdown("---")
st.sidebar.header("Analysis Scope")

@st.cache_data
def load_filter_options():
    try:
        engine = db_manager.connect()
        stores = pd.read_sql("SELECT DISTINCT store_id FROM forecasts ORDER BY store_id", engine)
        depts = pd.read_sql("SELECT DISTINCT dept_id FROM forecasts ORDER BY dept_id", engine)
        db_manager.close()
        return stores['store_id'].tolist(), depts['dept_id'].tolist()
    except Exception as e:
        logger.error("Failed to load sidebar filter options: %s", e, exc_info=True)
        return [], []

available_stores, available_depts = load_filter_options()

if not available_stores and not available_depts:
    st.sidebar.error("Could not load filter options — database may be unavailable.")

selected_store = st.sidebar.selectbox("Store", ["All"] + available_stores)
selected_dept = st.sidebar.selectbox("Department", ["All"] + available_depts)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_forecasts(store_filter, dept_filter):
    engine = db_manager.connect()
    query = """
        SELECT store_id, dept_id, forecast_date, predicted_sales,
               prediction_lower AS lower_bound, prediction_upper AS upper_bound,
               model_name, confidence_score
        FROM forecasts ORDER BY forecast_date ASC
    """
    df = pd.read_sql(query, engine)
    db_manager.close()
    if store_filter != "All":
        df = df[df['store_id'] == store_filter]
    if dept_filter != "All":
        df = df[df['dept_id'] == dept_filter]
    return df

@st.cache_data(ttl=300)
def load_historical_data(store_filter, dept_filter):
    engine = db_manager.connect()
    conditions = []
    if store_filter != "All":
        conditions.append(f"store_id = {store_filter}")
    if dept_filter != "All":
        conditions.append(f"dept_id = {dept_filter}")
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT store_id, dept_id, feature_date, weekly_sales, is_holiday,
               temperature, cpi, unemployment
        FROM engineered_features {where_clause}
        ORDER BY feature_date DESC
    """
    df = pd.read_sql(query, engine)
    db_manager.close()
    return df


# ------------------------------------------------------------------
# Helper: Render a structured AgentResponse
# ------------------------------------------------------------------
def render_agent_response(result):
    """
    Render an AgentResponse (or backward-compatible dict) in Streamlit.
    Shows: summary → critical alerts → insights → recommendations → full response.
    """
    # Handle both AgentResponse objects and raw dicts (backward compat)
    if isinstance(result, AgentResponse):
        _render_structured(result)
    elif isinstance(result, dict) and "response" in result:
        # Old-style raw dict — just show the response
        st.markdown(result["response"])
    else:
        st.warning("Unexpected response format from agent.")


def _render_structured(resp: AgentResponse):
    """Render a structured AgentResponse with layered detail."""

    # --- Status banner ---
    if resp.status == AgentStatus.FAILURE:
        st.error(f"**Agent failed:** {resp.error_message}")
        return
    elif resp.status == AgentStatus.PARTIAL:
        st.warning(f"**Partial result:** {resp.error_message}")

    # --- Summary (the key improvement — always visible) ---
    st.info(f"**Summary:** {resp.summary}")

    # --- Critical insights as red alerts ---
    critical = [i for i in resp.insights if i.severity == InsightSeverity.CRITICAL]
    for insight in critical:
        st.error(f"🚨 **{insight.title}:** {insight.detail}")

    # --- Insights cards ---
    non_critical = [i for i in resp.insights if i.severity != InsightSeverity.CRITICAL]
    if non_critical:
        st.markdown("**Key Insights:**")
        for insight in non_critical:
            icon = "⚠️" if insight.severity == InsightSeverity.WARNING else "ℹ️"
            with st.container():
                st.markdown(
                    f"{icon} **{insight.title}** — {insight.detail}"
                    + (f" *(metric: {insight.metric_value})*" if insight.metric_value is not None else "")
                )

    # --- Recommendations ---
    if resp.recommendations:
        st.markdown("**Recommendations:**")
        for i, rec in enumerate(resp.recommendations, 1):
            st.markdown(f"{i}. {rec}")

    # --- Full LLM response in expandable section ---
    if resp.llm_response:
        with st.expander("📄 Show detailed AI analysis", expanded=False):
            st.markdown(resp.llm_response)

    # --- Metadata footer ---
    if resp.metadata.get("execution_time_seconds"):
        st.caption(
            f"⏱ {resp.metadata['execution_time_seconds']}s | "
            f"Model: {resp.metadata.get('model', 'N/A')} | "
            f"Scope: {resp.metadata.get('context_summary', 'N/A')}"
        )


# ------------------------------------------------------------------
# Main content
# ------------------------------------------------------------------
st.markdown("### 💬 Ask the AI")

user_question = st.text_area(
    "What would you like to know?",
    placeholder="Example: What are the key drivers of sales for this store? How should I optimize inventory levels?",
    height=100,
)

analyze_button = st.button("🚀 Get AI Insights", type="primary")

if analyze_button and user_question:
    with st.spinner("🤖 AI agents are analyzing your data..."):
        forecasts = load_forecasts(selected_store, selected_dept)
        historical_data = load_historical_data(selected_store, selected_dept)

        if forecasts.empty:
            st.error(
                "⚠️ No forecasts found in the database. "
                "Please run `python models/generate_forecasts.py` first."
            )
            st.stop()

        store_val = selected_store if selected_store != "All" else None
        dept_val = selected_dept if selected_dept != "All" else None

        # ----------------------------------------------------------
        # Single agent modes: use safe_process for structured output
        # ----------------------------------------------------------
        if agent_type == "📊 Demand Forecasting":
            st.markdown("### 📊 Demand Forecasting Agent")

            context = {
                "forecasts": forecasts,
                "historical_sales": historical_data,
                "store_id": store_val,
                "dept_id": dept_val,
                "question": user_question,
            }
            result = orchestrator.ask_agent("demand", user_question, context)
            render_agent_response(result)

        elif agent_type == "📦 Inventory Optimization":
            st.markdown("### 📦 Inventory Optimization Agent")

            context = {
                "forecasts": forecasts,
                "service_level": 0.95,
                "lead_time_days": 7,
                "store_id": store_val,
                "dept_id": dept_val,
                "question": user_question,
            }
            result = orchestrator.ask_agent("inventory", user_question, context)
            render_agent_response(result)

        elif agent_type == "⚠️ Anomaly Detection":
            st.markdown("### ⚠️ Anomaly Detection Agent")

            result = orchestrator.get_agent("anomaly").detect_anomalies(
                historical_data, threshold=3.0
            )
            render_agent_response(result)

        # ----------------------------------------------------------
        # Comprehensive mode: all agents
        # ----------------------------------------------------------
        else:
            st.markdown("### 🔄 Comprehensive Multi-Agent Analysis")

            results = orchestrator.analyze_forecast(
                forecasts=forecasts,
                historical_sales=historical_data,
                store_id=store_val,
                dept_id=dept_val,
            )

            # Overall summary
            st.info(results["summary"])

            # Critical alerts banner
            if results.get("critical_alerts"):
                for alert in results["critical_alerts"]:
                    st.error(
                        f"🚨 **[{alert['agent'].upper()}] {alert['title']}:** "
                        f"{alert['detail']}"
                    )

            # ----------------------------------------------------------
            # Cross-agent synthesis (the key differentiator)
            # ----------------------------------------------------------
            cross = results.get("cross_agent_synthesis", [])
            if cross:
                st.markdown("### 🔗 Cross-Agent Insights")
                st.caption(
                    "These insights combine findings across agents — "
                    "connections and contradictions no single agent would catch alone."
                )

                for item in cross:
                    icon_map = {
                        "contradiction": "⚡",
                        "dependency": "🔗",
                        "reinforcement": "🔄",
                        "warning": "⚠️",
                    }
                    icon = icon_map.get(item["type"], "💡")
                    agents_str = " + ".join(
                        a.title() for a in item.get("agents", [])
                    )

                    with st.container():
                        st.markdown(
                            f"{icon} **{item['title']}** "
                            f"*({agents_str})*"
                        )
                        st.markdown(f"> {item['detail']}")
                        if item.get("action"):
                            st.markdown(f"**→ Action:** {item['action']}")
                        st.markdown("---")

            # Agent tabs
            tab1, tab2, tab3 = st.tabs([
                "📊 Demand Analysis",
                "📦 Inventory Recommendations",
                "⚠️ Anomaly Detection",
            ])

            raw = results.get("raw_responses", {})

            with tab1:
                st.markdown("#### Demand Forecasting Insights")
                if "demand" in raw:
                    _render_structured(raw["demand"])
                else:
                    st.warning("Demand agent did not return results.")

            with tab2:
                st.markdown("#### Inventory Optimization Recommendations")
                if "inventory" in raw:
                    _render_structured(raw["inventory"])
                else:
                    st.warning("Inventory agent did not return results.")

            with tab3:
                st.markdown("#### Anomaly Detection Report")
                if "anomaly" in raw:
                    _render_structured(raw["anomaly"])
                else:
                    st.warning("Anomaly agent did not return results.")

elif analyze_button and not user_question:
    st.warning("⚠️ Please enter a question to get AI insights")


# ------------------------------------------------------------------
# Example questions
# ------------------------------------------------------------------
st.markdown("---")
st.markdown("### 💡 Example Questions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Demand Forecasting:**
    - What are the key sales drivers for this period?
    - How will holidays impact sales?
    - What seasonal patterns should I expect?
    """)
    st.markdown("""
    **Inventory Optimization:**
    - What inventory levels should I maintain?
    - When should I reorder stock?
    - How much safety stock do I need?
    """)

with col2:
    st.markdown("""
    **Anomaly Detection:**
    - Are there any unusual sales patterns?
    - Which stores/departments show anomalies?
    - What might be causing sales spikes/drops?
    """)
    st.markdown("""
    **Comprehensive Analysis:**
    - Give me a complete analysis of this store
    - What actions should I take this week?
    - How can I improve performance?
    """)

st.markdown("---")
st.info("🤖 **Powered by Google Gemini** — Our AI agents use advanced language models to provide actionable insights")