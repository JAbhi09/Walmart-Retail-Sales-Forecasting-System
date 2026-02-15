"""
Forecast Visualization Page
Interactive sales forecast visualization and analysis
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from database.db_manager import db_manager

st.set_page_config(page_title="Forecast Visualization", page_icon="📊", layout="wide")

st.title("📊 Sales Forecast Visualization")
st.markdown("Explore 8-week sales predictions with interactive charts and filters")

# Sidebar filters
st.sidebar.header("Filters")


@st.cache_data
def load_filter_options():
    engine = db_manager.connect()
    stores = pd.read_sql("SELECT DISTINCT store_id FROM stores ORDER BY store_id", engine)
    depts = pd.read_sql("SELECT DISTINCT dept_id FROM raw_sales ORDER BY dept_id", engine)
    db_manager.close()
    return stores['store_id'].tolist(), depts['dept_id'].tolist()


stores, depts = load_filter_options()
selected_store = st.sidebar.selectbox("Select Store", ["All"] + stores)
selected_dept = st.sidebar.selectbox("Select Department", ["All"] + depts)


# ============================================================
# FIX: Load REAL forecasts from the forecasts table
# ============================================================
@st.cache_data(ttl=300)
def load_forecasts(store_filter, dept_filter):
    """Load actual model forecasts from the forecasts table"""
    engine = db_manager.connect()

    query = """
        SELECT
            store_id, dept_id, forecast_date,
            predicted_sales, prediction_lower AS lower_bound,
            prediction_upper AS upper_bound,
            model_name, confidence_score
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
    """Load historical data pre-aggregated by week for charting"""
    engine = db_manager.connect()

    # Build WHERE clause based on filters
    conditions = []
    if store_filter != "All":
        conditions.append(f"store_id = {store_filter}")
    if dept_filter != "All":
        conditions.append(f"dept_id = {dept_filter}")

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    # Pre-aggregate in SQL to avoid loading millions of rows
    query = f"""
        SELECT
            feature_date,
            SUM(weekly_sales) AS weekly_sales,
            MAX(CAST(is_holiday AS INTEGER)) AS is_holiday,
            COUNT(DISTINCT store_id) AS store_count,
            COUNT(DISTINCT dept_id) AS dept_count
        FROM engineered_features
        {where_clause}
        GROUP BY feature_date
        ORDER BY feature_date
    """

    df = pd.read_sql(query, engine)
    db_manager.close()

    return df


with st.spinner("Loading data..."):
    forecasts = load_forecasts(selected_store, selected_dept)
    historical_data = load_historical_data(selected_store, selected_dept)

# Check if forecasts exist
if forecasts.empty:
    st.error(
        "⚠️ No forecasts found in the database. "
        "Run `python models/generate_forecasts.py` first to generate predictions."
    )
    st.stop()

# Display metrics
st.markdown("### 📈 Forecast Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_forecast = forecasts['predicted_sales'].sum()
    st.metric("Total Forecasted Sales (8 weeks)", f"${total_forecast:,.0f}")

with col2:
    # Average total sales per forecast week (across all stores/depts)
    avg_weekly = forecasts.groupby('forecast_date')['predicted_sales'].sum().mean()
    st.metric("Avg Weekly Sales (All Stores)", f"${avg_weekly:,.0f}")

with col3:
    forecast_weeks = forecasts['forecast_date'].nunique()
    st.metric("Forecast Weeks", int(forecast_weeks))

with col4:
    if not historical_data.empty:
        # Compare forecast avg per week vs historical avg per week
        hist_avg = historical_data['weekly_sales'].mean()
        forecast_total_per_week = forecasts.groupby('forecast_date')['predicted_sales'].sum().mean()
        growth = ((forecast_total_per_week / hist_avg) - 1) * 100 if hist_avg > 0 else 0
        st.metric("vs Historical Avg", f"{growth:+.1f}%")
    else:
        st.metric("vs Historical Avg", "N/A")

st.markdown("---")

# Visualization tabs
tab1, tab2, tab3 = st.tabs(["📈 Forecast Chart", "📊 Historical Trends", "📋 Data Table"])

with tab1:
    st.markdown("### 8-Week Sales Forecast")

    # Historical data is already aggregated by date
    hist_agg = historical_data.sort_values('feature_date').tail(20)

    forecast_agg = forecasts.groupby('forecast_date').agg(
        predicted_sales=('predicted_sales', 'sum'),
        lower_bound=('lower_bound', 'sum'),
        upper_bound=('upper_bound', 'sum'),
    ).reset_index().sort_values('forecast_date')

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_agg['feature_date'],
        y=hist_agg['weekly_sales'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#0071ce', width=2),
        marker=dict(size=6)
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_agg['forecast_date'],
        y=forecast_agg['predicted_sales'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff6b35', width=2, dash='dash'),
        marker=dict(size=8)
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_agg['forecast_date'].tolist() + forecast_agg['forecast_date'].tolist()[::-1],
        y=forecast_agg['upper_bound'].tolist() + forecast_agg['lower_bound'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 107, 53, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True
    ))

    fig.update_layout(
        title="Sales Forecast with Confidence Interval",
        xaxis_title="Date",
        yaxis_title="Weekly Sales ($)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Historical Sales Trends")

    # Historical data is already aggregated
    full_hist_agg = historical_data.sort_values('feature_date')

    fig2 = px.line(
        full_hist_agg, x='feature_date', y='weekly_sales',
        title='Historical Weekly Sales',
        labels={'feature_date': 'Date', 'weekly_sales': 'Total Sales ($)'}
    )

    holiday_dates = full_hist_agg[full_hist_agg['is_holiday'] == True]
    fig2.add_trace(go.Scatter(
        x=holiday_dates['feature_date'],
        y=holiday_dates['weekly_sales'],
        mode='markers',
        name='Holiday Week',
        marker=dict(size=12, color='red', symbol='star')
    ))

    fig2.update_layout(height=500, template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Forecast Data Table")

    display_df = forecasts[[
        'forecast_date', 'store_id', 'dept_id',
        'predicted_sales', 'lower_bound', 'upper_bound', 'model_name'
    ]].copy()

    display_df['predicted_sales'] = display_df['predicted_sales'].apply(lambda x: f"${x:,.2f}")
    display_df['lower_bound'] = display_df['lower_bound'].apply(lambda x: f"${x:,.2f}")
    display_df['upper_bound'] = display_df['upper_bound'].apply(lambda x: f"${x:,.2f}")
    display_df.columns = ['Date', 'Store', 'Dept', 'Forecast', 'Lower Bound', 'Upper Bound', 'Model']

    st.dataframe(display_df, use_container_width=True, height=400)

    csv = forecasts.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name=f"forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

st.markdown("---")
st.info("💡 **Tip**: Use the filters in the sidebar to focus on specific stores or departments")