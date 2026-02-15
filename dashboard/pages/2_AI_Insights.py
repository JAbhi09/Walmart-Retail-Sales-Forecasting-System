"""
AI Insights Page
Interactive interface for multi-agent AI system
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from database.db_manager import db_manager
from agents.orchestrator import AgentOrchestrator

st.set_page_config(page_title="AI Insights", page_icon="🤖", layout="wide")

st.title("🤖 AI-Powered Insights")
st.markdown("Get intelligent recommendations from our multi-agent AI system")

# Initialize orchestrator
@st.cache_resource
def get_orchestrator():
    return AgentOrchestrator()

orchestrator = get_orchestrator()

# Sidebar - Agent selection
st.sidebar.header("AI Agent Selection")
agent_type = st.sidebar.radio(
    "Choose an agent:",
    ["📊 Demand Forecasting", "📦 Inventory Optimization", "⚠️ Anomaly Detection", "🔄 All Agents (Comprehensive)"]
)

st.sidebar.markdown("---")
st.sidebar.header("Analysis Scope")

@st.cache_data
def load_filter_options():
    engine = db_manager.connect()
    stores = pd.read_sql("SELECT DISTINCT store_id FROM forecasts ORDER BY store_id", engine)
    depts = pd.read_sql("SELECT DISTINCT dept_id FROM forecasts ORDER BY dept_id", engine)
    db_manager.close()
    return stores['store_id'].tolist(), depts['dept_id'].tolist()

available_stores, available_depts = load_filter_options()
selected_store = st.sidebar.selectbox("Store", ["All"] + available_stores)
selected_dept = st.sidebar.selectbox("Department", ["All"] + available_depts)


# ============================================================
# FIX: Load REAL forecasts from database instead of faking them
# ============================================================
@st.cache_data(ttl=300)
def load_forecasts(store_filter, dept_filter):
    """Load actual model forecasts from the forecasts table"""
    engine = db_manager.connect()

    query = """
        SELECT
            store_id,
            dept_id,
            forecast_date,
            predicted_sales,
            prediction_lower AS lower_bound,
            prediction_upper AS upper_bound,
            model_name,
            confidence_score
        FROM forecasts
        ORDER BY forecast_date ASC
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
    """Load historical sales data for context"""
    engine = db_manager.connect()

    # Build WHERE clause based on filters
    conditions = []
    if store_filter != "All":
        conditions.append(f"store_id = {store_filter}")
    if dept_filter != "All":
        conditions.append(f"dept_id = {dept_filter}")

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    query = f"""
        SELECT
            store_id, dept_id, feature_date, weekly_sales, is_holiday,
            temperature, cpi, unemployment
        FROM engineered_features
        {where_clause}
        ORDER BY feature_date DESC
    """

    df = pd.read_sql(query, engine)
    db_manager.close()

    return df


# Main content
st.markdown("### 💬 Ask the AI")

user_question = st.text_area(
    "What would you like to know?",
    placeholder="Example: What are the key drivers of sales for this store? How should I optimize inventory levels?",
    height=100
)

analyze_button = st.button("🚀 Get AI Insights", type="primary")

if analyze_button and user_question:
    with st.spinner("🤖 AI agents are analyzing your data..."):
        # Load REAL data
        forecasts = load_forecasts(selected_store, selected_dept)
        historical_data = load_historical_data(selected_store, selected_dept)

        # Check if forecasts exist
        if forecasts.empty:
            st.error(
                "⚠️ No forecasts found in the database. "
                "Please run `python models/generate_forecasts.py` first."
            )
            st.stop()

        if agent_type == "📊 Demand Forecasting":
            st.markdown("### 📊 Demand Forecasting Agent Response")

            context = {
                'forecasts': forecasts,
                'historical_sales': historical_data,
                'store_id': selected_store if selected_store != "All" else None,
                'dept_id': selected_dept if selected_dept != "All" else None,
                'question': user_question
            }

            result = orchestrator.ask_agent('demand', user_question, context)
            st.markdown(result['response'])

        elif agent_type == "📦 Inventory Optimization":
            st.markdown("### 📦 Inventory Optimization Agent Response")

            context = {
                'forecasts': forecasts,
                'service_level': 0.95,
                'lead_time_days': 7,
                'store_id': selected_store if selected_store != "All" else None,
                'dept_id': selected_dept if selected_dept != "All" else None,
                'question': user_question
            }

            result = orchestrator.ask_agent('inventory', user_question, context)
            st.markdown(result['response'])

        elif agent_type == "⚠️ Anomaly Detection":
            st.markdown("### ⚠️ Anomaly Detection Agent Response")

            result = orchestrator.get_agent('anomaly').detect_anomalies(
                historical_data, threshold=3.0
            )
            st.markdown(result['response'])

        else:  # All Agents
            st.markdown("### 🔄 Comprehensive Multi-Agent Analysis")

            results = orchestrator.analyze_forecast(
                forecasts=forecasts,
                historical_sales=historical_data,
                store_id=selected_store if selected_store != "All" else None,
                dept_id=selected_dept if selected_dept != "All" else None
            )

            st.info(results['summary'])

            tab1, tab2, tab3 = st.tabs([
                "📊 Demand Analysis",
                "📦 Inventory Recommendations",
                "⚠️ Anomaly Detection"
            ])

            with tab1:
                st.markdown("#### Demand Forecasting Insights")
                st.markdown(results['detailed_insights']['demand_analysis']['response'])

            with tab2:
                st.markdown("#### Inventory Optimization Recommendations")
                st.markdown(results['detailed_insights']['inventory_recommendations']['response'])

            with tab3:
                st.markdown("#### Anomaly Detection Report")
                st.markdown(results['detailed_insights']['anomaly_detection']['response'])

elif analyze_button and not user_question:
    st.warning("⚠️ Please enter a question to get AI insights")

# Example questions
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
st.info("🤖 **Powered by Google Gemini** - Our AI agents use advanced language models to provide actionable insights")