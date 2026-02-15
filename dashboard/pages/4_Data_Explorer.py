"""
Data Explorer Page
Interactive data exploration and analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from database.db_manager import db_manager

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")

st.title("🔍 Data Explorer")
st.markdown("Explore and analyze historical sales data")

# Sidebar filters
st.sidebar.header("Data Filters")

# Date range
date_range = st.sidebar.date_input(
    "Date Range",
    value=(pd.to_datetime("2012-01-01"), pd.to_datetime("2012-10-26"))
)

# Store and department filters
@st.cache_data
def load_filter_options():
    engine = db_manager.connect()
    stores = pd.read_sql("SELECT DISTINCT store_id FROM stores ORDER BY store_id", engine)
    depts = pd.read_sql("SELECT DISTINCT dept_id FROM raw_sales ORDER BY dept_id LIMIT 50", engine)
    db_manager.close()
    return stores['store_id'].tolist(), depts['dept_id'].tolist()

stores, depts = load_filter_options()

selected_stores = st.sidebar.multiselect("Stores", stores, default=stores[:5])
selected_depts = st.sidebar.multiselect("Departments", depts, default=depts[:5])

# Load data
@st.cache_data(ttl=300)
def load_sales_data(start_date, end_date, store_list, dept_list):
    engine = db_manager.connect()
    
    # Build query with filters
    store_filter = f"AND store_id IN ({','.join(map(str, store_list))})" if store_list else ""
    dept_filter = f"AND dept_id IN ({','.join(map(str, dept_list))})" if dept_list else ""
    
    query = f"""
        SELECT 
            feature_date,
            store_id,
            dept_id,
            weekly_sales,
            is_holiday,
            temperature,
            fuel_price,
            cpi,
            unemployment
        FROM engineered_features
        WHERE feature_date BETWEEN '{start_date}' AND '{end_date}'
        {store_filter}
        {dept_filter}
        ORDER BY feature_date DESC
        LIMIT 10000
    """
    
    df = pd.read_sql(query, engine)
    db_manager.close()
    
    return df

if selected_stores and selected_depts:
    with st.spinner("Loading data..."):
        data = load_sales_data(date_range[0], date_range[1], selected_stores, selected_depts)
    
    # Summary statistics
    st.markdown("### 📊 Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    
    with col2:
        st.metric("Total Sales", f"${data['weekly_sales'].sum():,.2f}")
    
    with col3:
        st.metric("Avg Weekly Sales", f"${data['weekly_sales'].mean():,.2f}")
    
    with col4:
        st.metric("Date Range", f"{len(data['feature_date'].unique())} weeks")
    
    st.markdown("---")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "📊 Distributions", "🔥 Heatmap", "📋 Raw Data"])
    
    with tab1:
        st.markdown("### Sales Over Time")
        
        # Aggregate by date
        time_series = data.groupby('feature_date')['weekly_sales'].sum().reset_index()
        
        fig = px.line(
            time_series,
            x='feature_date',
            y='weekly_sales',
            title='Total Weekly Sales Over Time',
            labels={'feature_date': 'Date', 'weekly_sales': 'Total Sales ($)'}
        )
        
        # Add holiday markers
        holiday_data = data[data['is_holiday'] == True].groupby('feature_date')['weekly_sales'].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=holiday_data['feature_date'],
            y=holiday_data['weekly_sales'],
            mode='markers',
            name='Holiday Week',
            marker=dict(size=12, color='red', symbol='star')
        ))
        
        fig.update_layout(height=500, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # By store
        st.markdown("#### Sales by Store")
        store_series = data.groupby(['feature_date', 'store_id'])['weekly_sales'].sum().reset_index()
        
        fig2 = px.line(
            store_series,
            x='feature_date',
            y='weekly_sales',
            color='store_id',
            title='Weekly Sales by Store',
            labels={'feature_date': 'Date', 'weekly_sales': 'Sales ($)', 'store_id': 'Store'}
        )
        
        fig2.update_layout(height=500, template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.markdown("### Sales Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = px.histogram(
                data,
                x='weekly_sales',
                nbins=50,
                title='Sales Distribution',
                labels={'weekly_sales': 'Weekly Sales ($)'}
            )
            fig3.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig4 = px.box(
                data,
                y='weekly_sales',
                x='is_holiday',
                title='Sales: Holiday vs Non-Holiday',
                labels={'weekly_sales': 'Weekly Sales ($)', 'is_holiday': 'Holiday Week'}
            )
            fig4.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig4, use_container_width=True)
        
        # By store
        st.markdown("#### Sales by Store")
        fig5 = px.box(
            data,
            y='weekly_sales',
            x='store_id',
            title='Sales Distribution by Store',
            labels={'weekly_sales': 'Weekly Sales ($)', 'store_id': 'Store'}
        )
        fig5.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig5, use_container_width=True)
    
    with tab3:
        st.markdown("### Sales Heatmap")
        
        # Create pivot table
        pivot_data = data.groupby(['store_id', 'dept_id'])['weekly_sales'].mean().reset_index()
        pivot_table = pivot_data.pivot(index='dept_id', columns='store_id', values='weekly_sales')
        
        fig6 = px.imshow(
            pivot_table,
            title='Average Sales by Store and Department',
            labels=dict(x='Store', y='Department', color='Avg Sales ($)'),
            color_continuous_scale='Blues'
        )
        
        fig6.update_layout(height=600)
        st.plotly_chart(fig6, use_container_width=True)
    
    with tab4:
        st.markdown("### Raw Data")
        
        # Display options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search = st.text_input("🔍 Search", placeholder="Filter data...")
        
        with col2:
            rows_to_show = st.selectbox("Rows", [10, 25, 50, 100], index=1)
        
        # Filter data
        display_data = data.copy()
        if search:
            display_data = display_data[
                display_data.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
            ]
        
        # Format for display
        display_data['weekly_sales'] = display_data['weekly_sales'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(display_data.head(rows_to_show), use_container_width=True, height=400)
        
        # Download button
        csv = data.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset CSV",
            data=csv,
            file_name=f"sales_data_{date_range[0]}_{date_range[1]}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # Correlation analysis
    st.markdown("### 🔗 Correlation Analysis")
    
    numeric_cols = ['weekly_sales', 'temperature', 'fuel_price', 'cpi', 'unemployment']
    corr_data = data[numeric_cols].corr()
    
    fig7 = px.imshow(
        corr_data,
        title='Feature Correlation Matrix',
        labels=dict(color='Correlation'),
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    
    fig7.update_layout(height=500)
    st.plotly_chart(fig7, use_container_width=True)

else:
    st.warning("⚠️ Please select at least one store and one department to explore data")

st.markdown("---")
st.info("💡 **Tip**: Use the sidebar filters to focus your analysis on specific stores, departments, or time periods")
