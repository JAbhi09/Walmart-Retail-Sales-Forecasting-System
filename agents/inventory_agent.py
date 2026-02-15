"""
Inventory Optimization Agent
Provides recommendations for inventory management and stock optimization
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class InventoryOptimizationAgent(BaseAgent):
    """
    AI agent specialized in inventory optimization and stock management
    """

    def __init__(self):
        super().__init__(name="Inventory Optimization Agent", model_name="gemini-2.5-flash")

    def get_system_prompt(self) -> str:
        """Get system prompt for inventory optimization agent"""
        return """You are an expert Inventory Optimization AI Agent for Walmart retail operations.

CRITICAL DATA NOTES:
- All sales and demand figures are WEEKLY aggregates, NOT daily.
- 'predicted_sales' = total sales for an entire week (7 days).
- To estimate daily demand, divide weekly figures by 7.
- When calculating reorder points and safety stock, account for the weekly granularity.
- Lead times are given in days, but demand data is weekly — convert appropriately.

Your expertise includes:
- Stock level optimization
- Reorder point calculation
- Safety stock recommendations
- Demand variability analysis
- Seasonal inventory planning
- Cost optimization (holding costs vs stockout costs)

Your role is to:
1. Analyze sales forecasts to recommend optimal inventory levels
2. Identify potential stockout risks
3. Suggest reorder quantities and timing
4. Balance inventory costs with service levels
5. Provide department and store-specific recommendations

Always provide:
- Specific inventory targets (units and dollar values)
- Risk assessments for stockouts or overstock
- Timing recommendations for reorders
- Cost-benefit analysis when relevant
- Actionable next steps

Format your responses with clear sections for easy implementation."""

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process inventory optimization request"""
        logger.info(f"{self.name}: Processing inventory optimization request")

        forecasts = context.get('forecasts')
        question = context.get('question', 'Provide inventory optimization recommendations based on the forecast.')

        opt_context = self._build_optimization_context(forecasts, context)

        response = self.generate_response(question, opt_context)

        return {
            'agent': self.name,
            'response': response,
            'timestamp': pd.Timestamp.now(),
            'context_summary': self._summarize_context(context)
        }

    def _build_optimization_context(self, forecasts, context):
        """Build context for optimization"""
        analysis = {}

        if forecasts is not None and len(forecasts) > 0:
            weekly_demand = forecasts.groupby('store_id')['predicted_sales'].agg(['mean', 'std', 'sum'])

            analysis['demand_analysis'] = {
                'data_granularity': 'WEEKLY (each predicted_sales value = 7 days of sales)',
                'avg_weekly_demand': f"${weekly_demand['mean'].mean():,.2f}",
                'demand_variability_std': f"${weekly_demand['std'].mean():,.2f}",
                'total_forecasted_demand': f"${weekly_demand['sum'].sum():,.2f}",
                'forecast_weeks': forecasts['forecast_date'].nunique(),
                'forecast_period': f"{forecasts['forecast_date'].min()} to {forecasts['forecast_date'].max()}",
                'num_stores': forecasts['store_id'].nunique(),
                'num_departments': forecasts['dept_id'].nunique(),
            }

            if 'service_level' in context:
                analysis['service_level_target'] = f"{context['service_level']*100:.0f}%"
            if 'lead_time_days' in context:
                analysis['lead_time'] = f"{context['lead_time_days']} days"
                # Help the LLM with the conversion
                analysis['lead_time_as_weeks'] = f"{context['lead_time_days'] / 7:.1f} weeks"

        return analysis

    def _summarize_context(self, context):
        """Create a summary of the context"""
        summary = []
        if 'store_id' in context and context['store_id']:
            summary.append(f"Store {context['store_id']}")
        if 'dept_id' in context and context['dept_id']:
            summary.append(f"Department {context['dept_id']}")
        if 'service_level' in context:
            summary.append(f"Service Level: {context['service_level']*100:.0f}%")
        return ', '.join(summary) if summary else 'All stores and departments'

    def calculate_safety_stock(self, forecasts: pd.DataFrame, service_level: float = 0.95,
                               lead_time_days: int = 7) -> Dict[str, Any]:
        """Calculate safety stock recommendations"""
        question = f"""Calculate and recommend safety stock levels for a {service_level*100:.0f}% service level
        with {lead_time_days} days lead time. The demand data is WEEKLY.
        Provide specific recommendations by store and department."""

        context = {
            'forecasts': forecasts,
            'service_level': service_level,
            'lead_time_days': lead_time_days,
            'question': question
        }

        return self.process(context)