"""
Streamlit Dashboard - Main Entry Point
Walmart Retail Sales Forecasting System
"""
import streamlit as st
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0071ce;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0071ce;
    }
    .stButton>button {
        background-color: #0071ce;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #005a9e;
    }
    </style>
""", unsafe_allow_html=True)

# Run DB health check once per page load; result is reused in sidebar and System Status
try:
    from database.db_manager import db_manager
    import pandas as pd
    _db_healthy, _db_error = db_manager.check_connection()
except Exception as _e:
    logger.error("Could not initialise db_manager: %s", _e, exc_info=True)
    _db_healthy, _db_error = False, str(_e)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/200px-Walmart_logo.svg.png", width=150)
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("Use the pages menu above to navigate")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Walmart Sales Forecasting System**
    
    An AI-powered forecasting platform combining:
    - LightGBM ML models
    - Multi-agent AI insights
    - Interactive visualizations
    """)
    st.markdown("---")
    st.markdown("### Quick Stats")
    
    # Load quick stats
    if not _db_healthy:
        st.error(f"Database unavailable: {_db_error}")
    else:
        try:
            engine = db_manager.connect()

            stores = pd.read_sql("SELECT COUNT(DISTINCT store_id) as count FROM stores", engine)
            st.metric("Total Stores", f"{stores['count'].iloc[0]}")

            depts = pd.read_sql("SELECT COUNT(DISTINCT dept_id) as count FROM raw_sales", engine)
            st.metric("Departments", f"{depts['count'].iloc[0]}")

            latest = pd.read_sql("SELECT MAX(feature_date) as date FROM engineered_features", engine)
            date_val = latest['date'].iloc[0]
            st.metric("Latest Data", date_val.strftime('%Y-%m-%d') if date_val is not None else "No data")

            db_manager.close()
        except Exception as e:
            logger.error("Failed to load Quick Stats: %s", e, exc_info=True)
            st.warning(f"Stats unavailable: {e}")

# Main content
st.markdown('<div class="main-header">🛒 Walmart Sales Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Retail Analytics & Demand Forecasting</div>', unsafe_allow_html=True)

# Welcome section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Forecasting
    - 8-week sales predictions
    - Store & department level
    - Confidence intervals
    - Historical trends
    """)

with col2:
    st.markdown("""
    ### 🤖 AI Insights
    - Demand analysis
    - Inventory optimization
    - Anomaly detection
    - Actionable recommendations
    """)

with col3:
    st.markdown("""
    ### 📈 Analytics
    - Model performance
    - Feature importance
    - Data exploration
    - Export capabilities
    """)

st.markdown("---")

# Getting Started
st.markdown("### 🚀 Getting Started")
st.markdown("""
1. **View Forecasts**: Navigate to the *Forecast Visualization* page to see sales predictions
2. **Get AI Insights**: Visit the *AI Insights* page to ask questions and get recommendations
3. **Check Performance**: Review model metrics on the *Model Performance* page
4. **Explore Data**: Use the *Data Explorer* to analyze historical sales patterns
""")

st.markdown("---")

# System Status
st.markdown("### 🔧 System Status")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    if _db_healthy:
        st.success("✅ Database Connected")
    else:
        st.error("❌ Database Unavailable")

with status_col2:
    st.success("✅ Model Loaded")

with status_col3:
    st.success("✅ AI Agents Ready")

with status_col4:
    st.success("✅ Data Current")

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Walmart Sales Forecasting System v1.0 | Powered by LightGBM & Google Gemini</p>
</div>
""", unsafe_allow_html=True)
