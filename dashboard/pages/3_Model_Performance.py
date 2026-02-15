"""
Model Performance Page
Display model metrics, feature importance, and evaluation results
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

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.title("📈 Model Performance Metrics")
st.markdown("Evaluate forecasting model accuracy and feature importance")

# Load model metadata
@st.cache_data(ttl=600)
def load_model_metadata():
    engine = db_manager.connect()
    
    query = """
        SELECT *
        FROM model_metadata
        ORDER BY training_date DESC
        LIMIT 1
    """
    
    df = pd.read_sql(query, engine)
    db_manager.close()
    
    return df

metadata = load_model_metadata()

if len(metadata) > 0:
    latest_model = metadata.iloc[0]
    
    # Model Overview
    st.markdown("### 🎯 Latest Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("WMAE", f"{latest_model['wmae']:.2f}", help="Weighted Mean Absolute Error (lower is better)")
    
    with col2:
        st.metric("MAE", f"{latest_model['mae']:.2f}", help="Mean Absolute Error")
    
    with col3:
        st.metric("RMSE", f"{latest_model['rmse']:.2f}", help="Root Mean Squared Error")
    
    with col4:
        baseline_wmae = 821.0
        improvement = ((baseline_wmae - latest_model['wmae']) / baseline_wmae) * 100
        st.metric("vs Baseline", f"{improvement:.1f}%", help=f"Improvement over baseline WMAE of {baseline_wmae}")
    
    st.markdown("---")
    
    # Model Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Model Information")
        st.write(f"**Model Name:** {latest_model['model_name']}")
        st.write(f"**Version:** {latest_model['model_version']}")
        st.write(f"**Training Date:** {latest_model['training_date']}")
        st.write(f"**MLflow Run ID:** `{latest_model['run_id']}`")
    
    with col2:
        st.markdown("### ⚙️ Hyperparameters")
        import json
        params = latest_model['parameters'] if isinstance(latest_model['parameters'], dict) else json.loads(latest_model['parameters'])
        for key, value in params.items():
            st.write(f"**{key}:** {value}")
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("### 🔍 Feature Importance Analysis")
    
    import json
    feature_importance = pd.DataFrame(latest_model['feature_importance'] if isinstance(latest_model['feature_importance'], list) else json.loads(latest_model['feature_importance']))
    feature_importance['importance'] = (feature_importance['importance'] / feature_importance['importance'].sum()) * 100
    # Top 20 features
    top_features = feature_importance.head(20)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 20 Most Important Features',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature categories
    st.markdown("### 📊 Feature Categories")
    
    # Categorize features
    def categorize_feature(feature_name):
        if 'lag' in feature_name:
            return 'Lag Features'
        elif 'rolling' in feature_name:
            return 'Rolling Statistics'
        elif any(x in feature_name for x in ['week', 'month', 'quarter']):
            return 'Temporal Features'
        elif any(x in feature_name for x in ['temperature', 'fuel', 'cpi', 'unemployment']):
            return 'Economic Indicators'
        elif 'markdown' in feature_name:
            return 'Markdown Features'
        elif 'store' in feature_name:
            return 'Store Features'
        else:
            return 'Other'
    
    feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)
    
    category_importance = feature_importance.groupby('category')['importance'].sum().reset_index()
    category_importance = category_importance.sort_values('importance', ascending=False)
    
    fig2 = px.pie(
        category_importance,
        values='importance',
        names='category',
        title='Feature Importance by Category',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Feature table
    st.markdown("### 📋 All Features")
    
    feature_display = feature_importance.copy()
    feature_display['importance'] = feature_display['importance'].apply(lambda x: f"{x:,.2f}")
    
    st.dataframe(feature_display, use_container_width=True, height=400)
    
    # Download button
    csv = feature_importance.to_csv(index=False)
    st.download_button(
        label="📥 Download Feature Importance CSV",
        data=csv,
        file_name="feature_importance.csv",
        mime="text/csv"
    )

else:
    st.warning("⚠️ No model metadata found. Please train a model first.")
    st.info("Run `python scripts/train_model.py` to train the forecasting model")

st.markdown("---")

# Performance Guidelines
st.markdown("### 📚 Performance Metrics Guide")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **WMAE (Weighted MAE)**
    - Primary competition metric
    - Holiday weeks weighted 5x
    - Lower is better
    - Target: < 821 (baseline)
    """)

with col2:
    st.markdown("""
    **MAE (Mean Absolute Error)**
    - Average prediction error
    - Easy to interpret ($)
    - Robust to outliers
    - Lower is better
    """)

with col3:
    st.markdown("""
    **RMSE (Root Mean Squared Error)**
    - Penalizes large errors
    - Same units as target
    - Sensitive to outliers
    - Lower is better
    """)
