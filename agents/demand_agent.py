"""
Demand Forecasting Agent
Provides intelligent insights and recommendations for sales forecasting
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DemandForecastingAgent(BaseAgent):
    """
    AI agent specialized in demand forecasting analysis and recommendations
    """

    def __init__(self):
        super().__init__(name="Demand Forecasting Agent", model_name="gemini-2.5-flash")

    def get_system_prompt(self) -> str:
        """Get system prompt for demand forecasting agent"""
        return """You are an expert Demand Forecasting AI Agent for Walmart retail operations.

CRITICAL DATA NOTES:
- All sales figures are WEEKLY aggregates, NOT daily. Each row = one week of sales.
- 'predicted_sales' values represent total sales for an entire week.
- When comparing forecasts to historical averages, compare weekly to weekly directly.
- Do NOT divide weekly figures by 7 to get daily estimates unless explicitly asked.
- Forecast dates are FUTURE dates beyond the training data. Historical dates are PAST data used for training.

Your expertise includes:
- Sales trend analysis and pattern recognition
- Seasonal demand forecasting
- Holiday impact assessment
- Store and department-level predictions
- Economic indicator interpretation

Your role is to:
1. Analyze sales forecasts and historical data
2. Identify trends, patterns, and anomalies
3. Provide actionable recommendations for inventory planning
4. Explain forecast drivers and confidence levels
5. Suggest strategies to optimize sales during key periods

Always provide:
- Clear, data-driven insights
- Specific, actionable recommendations
- Confidence levels for predictions
- Risk factors and mitigation strategies

Format your responses in a professional, concise manner suitable for retail managers."""

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process demand forecasting request
        """
        logger.info(f"{self.name}: Processing demand forecasting request")

        forecasts = context.get('forecasts')
        historical_sales = context.get('historical_sales')
        question = context.get('question', 'Analyze the demand forecast and provide insights.')

        analysis_context = self._build_analysis_context(forecasts, historical_sales, context)

        response = self.generate_response(question, analysis_context)

        return {
            'agent': self.name,
            'response': response,
            'timestamp': pd.Timestamp.now(),
            'context_summary': self._summarize_context(context)
        }

    def _build_analysis_context(self, forecasts, historical_sales, context):
        """Build context for analysis — clearly separates forecast vs historical"""
        analysis = {}

        # ============================================================
        # FIX: Use forecast_date (future) not feature_date (historical)
        # and clearly label everything as WEEKLY
        # ============================================================
        if forecasts is not None and len(forecasts) > 0:
            analysis['forecast_summary'] = {
                'data_granularity': 'WEEKLY (each value = one full week of sales)',
                'total_predicted_weekly_sales': f"${forecasts['predicted_sales'].sum():,.2f}",
                'avg_predicted_weekly_sales': f"${forecasts['predicted_sales'].mean():,.2f}",
                'min_predicted_weekly_sales': f"${forecasts['predicted_sales'].min():,.2f}",
                'max_predicted_weekly_sales': f"${forecasts['predicted_sales'].max():,.2f}",
                'forecast_period': f"{forecasts['forecast_date'].min()} to {forecasts['forecast_date'].max()}",
                'num_weeks_forecasted': forecasts['forecast_date'].nunique(),
                'num_store_dept_combinations': len(forecasts.groupby(['store_id', 'dept_id'])),
            }

            # Add per-store breakdown if multiple stores
            if forecasts['store_id'].nunique() <= 10:
                store_summary = forecasts.groupby('store_id')['predicted_sales'].agg(['mean', 'sum']).round(2)
                analysis['forecast_by_store'] = store_summary.to_dict()

        if historical_sales is not None and len(historical_sales) > 0:
            analysis['historical_summary'] = {
                'data_granularity': 'WEEKLY (each value = one full week of sales)',
                'avg_weekly_sales': f"${historical_sales['weekly_sales'].mean():,.2f}",
                'median_weekly_sales': f"${historical_sales['weekly_sales'].median():,.2f}",
                'std_weekly_sales': f"${historical_sales['weekly_sales'].std():,.2f}",
                'total_historical_sales': f"${historical_sales['weekly_sales'].sum():,.2f}",
                'historical_date_range': f"{historical_sales['feature_date'].min()} to {historical_sales['feature_date'].max()}",
                'num_historical_weeks': historical_sales['feature_date'].nunique(),
            }

            # Trend: compare recent 8 weeks vs prior 8 weeks
            sorted_hist = historical_sales.sort_values('feature_date', ascending=False)
            dates = sorted_hist['feature_date'].unique()
            if len(dates) >= 16:
                recent_avg = sorted_hist[sorted_hist['feature_date'].isin(dates[:8])]['weekly_sales'].mean()
                prior_avg = sorted_hist[sorted_hist['feature_date'].isin(dates[8:16])]['weekly_sales'].mean()
                trend_pct = ((recent_avg - prior_avg) / prior_avg) * 100
                analysis['recent_trend'] = {
                    'recent_8wk_avg': f"${recent_avg:,.2f}",
                    'prior_8wk_avg': f"${prior_avg:,.2f}",
                    'trend_change_pct': f"{trend_pct:+.1f}%"
                }

        if 'store_id' in context and context['store_id']:
            analysis['scope'] = f"Store {context['store_id']}"
        if 'dept_id' in context and context['dept_id']:
            analysis['scope'] = analysis.get('scope', '') + f", Department {context['dept_id']}"

        return analysis

    def _summarize_context(self, context):
        """Create a summary of the context"""
        summary = []
        if 'store_id' in context and context['store_id']:
            summary.append(f"Store {context['store_id']}")
        if 'dept_id' in context and context['dept_id']:
            summary.append(f"Department {context['dept_id']}")
        return ', '.join(summary) if summary else 'All stores and departments'

    def analyze_forecast_accuracy(self, actual: pd.DataFrame, predicted: pd.DataFrame) -> Dict[str, Any]:
        """Analyze forecast accuracy and provide insights"""
        question = """Analyze the forecast accuracy comparing actual vs predicted weekly sales.
        Identify where the model performs well and where it struggles.
        Provide specific recommendations to improve forecast accuracy."""

        merged = actual.merge(predicted, on=['store_id', 'dept_id', 'feature_date'], how='inner')
        mae = np.abs(merged['weekly_sales'] - merged['predicted_sales']).mean()
        mape = (np.abs(merged['weekly_sales'] - merged['predicted_sales']) / merged['weekly_sales']).mean() * 100

        context = {
            'actual_sales': actual,
            'predicted_sales': predicted,
            'accuracy_metrics': {
                'mae': f"${mae:,.2f}",
                'mape': f"{mape:.2f}%",
                'num_comparisons': len(merged)
            }
        }

        return self.process({**context, 'question': question})