"""
Anomaly Detection Agent
Identifies unusual patterns and potential issues in sales data
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AnomalyDetectionAgent(BaseAgent):
    """
    AI agent specialized in detecting anomalies and unusual patterns
    """
    
    def __init__(self):
        super().__init__(name="Anomaly Detection Agent", model_name="gemini-2.5-flash")
    
    def get_system_prompt(self) -> str:
        """Get system prompt for anomaly detection agent"""
        return """You are an expert Anomaly Detection AI Agent for Walmart retail operations.

        CRITICAL DATA NOTES:
- All sales figures are WEEKLY aggregates, NOT daily.

FORMATTING RULES:
- Use ### for main sections (not # or ##)
- Use #### for subsections
- Do NOT wrap dollar amounts in bold markers (**)
- Keep section headers concise (max 8 words)
- Use tables for data comparisons where possible

Your expertise includes:
- Statistical anomaly detection
- Pattern deviation analysis
- Trend break identification
- Unusual sales spike/drop detection
- Data quality issue identification
- Root cause analysis

Your role is to:
1. Identify unusual patterns in sales data
2. Distinguish between normal variation and true anomalies
3. Assess the severity and business impact of anomalies
4. Suggest potential root causes
5. Recommend investigation steps and corrective actions

Always provide:
- Clear identification of anomalies with specific data points
- Severity assessment (low/medium/high/critical)
- Potential business impact
- Recommended actions
- Timeline for investigation/resolution

Be precise with numbers and dates. Prioritize actionable insights."""
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process anomaly detection request
        
        Args:
            context: Dictionary containing:
                - sales_data: DataFrame with sales data
                - anomalies: List of detected anomalies
                - threshold: Anomaly detection threshold
                - question: User's specific question
        
        Returns:
            Dict with agent response and metadata
        """
        logger.info(f"{self.name}: Processing anomaly detection request")
        
        # Extract context
        sales_data = context.get('sales_data')
        anomalies = context.get('anomalies', [])
        question = context.get('question', 'Analyze the detected anomalies and provide insights.')
        
        # Build anomaly context
        anomaly_context = self._build_anomaly_context(sales_data, anomalies, context)
        
        # Generate response
        response = self.generate_response(question, anomaly_context)
        
        return {
            'agent': self.name,
            'response': response,
            'timestamp': pd.Timestamp.now(),
            'anomalies_detected': len(anomalies),
            'context_summary': self._summarize_context(context)
        }
    
    def _build_anomaly_context(self, sales_data, anomalies, context):
        """Build context for anomaly analysis"""
        analysis = {}
        
        if sales_data is not None and len(sales_data) > 0:
            analysis['data_summary'] = {
                'total_records': len(sales_data),
                'date_range': f"{sales_data['feature_date'].min()} to {sales_data['feature_date'].max()}",
                'avg_sales': f"${sales_data['weekly_sales'].mean():,.2f}",
                'sales_std': f"${sales_data['weekly_sales'].std():,.2f}"
            }
        
        if anomalies:
            analysis['anomalies_found'] = len(anomalies)
            analysis['anomaly_details'] = anomalies[:10]  # Top 10 anomalies
        
        if 'threshold' in context:
            analysis['detection_threshold'] = context['threshold']
        
        return analysis
    
    def _summarize_context(self, context):
        """Create a summary of the context"""
        summary = []
        if 'store_id' in context:
            summary.append(f"Store {context['store_id']}")
        if 'dept_id' in context:
            summary.append(f"Department {context['dept_id']}")
        return ', '.join(summary) if summary else 'All stores and departments'
    
    def detect_anomalies(self, sales_data: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect anomalies in sales data using statistical methods
        
        Args:
            sales_data: DataFrame with sales data
            threshold: Number of standard deviations for anomaly detection
        
        Returns:
            Dict with detected anomalies and analysis
        """
        logger.info(f"{self.name}: Detecting anomalies with threshold={threshold}")
        
        # Calculate z-scores
        sales_data = sales_data.copy()
        sales_data['z_score'] = np.abs((sales_data['weekly_sales'] - sales_data['weekly_sales'].mean()) / 
                                       sales_data['weekly_sales'].std())
        
        # Identify anomalies
        anomalies_df = sales_data[sales_data['z_score'] > threshold]
        
        # Convert to list of dicts
        anomalies = []
        for _, row in anomalies_df.head(20).iterrows():  # Top 20 anomalies
            anomalies.append({
                'date': str(row['feature_date']),
                'store_id': int(row['store_id']),
                'dept_id': int(row['dept_id']),
                'sales': f"${row['weekly_sales']:,.2f}",
                'z_score': f"{row['z_score']:.2f}",
                'deviation': f"{(row['z_score'] * sales_data['weekly_sales'].std()):,.2f}"
            })
        
        question = f"""Analyze these {len(anomalies_df)} detected anomalies. 
        Identify patterns, assess severity, and provide recommendations for investigation."""
        
        context = {
            'sales_data': sales_data,
            'anomalies': anomalies,
            'threshold': threshold,
            'question': question
        }
        
        return self.process(context)
