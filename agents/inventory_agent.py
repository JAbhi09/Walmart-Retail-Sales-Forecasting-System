"""
Inventory Optimization Agent (v2)
Overrides summary/insight hooks for structured responses.
All existing functionality preserved.
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List
import logging

from agents.base_agent import BaseAgent
from agents.response_model import Insight, InsightSeverity
from agents.validation import validate_dataframe

logger = logging.getLogger(__name__)


class InventoryOptimizationAgent(BaseAgent):
    """
    AI agent specialized in inventory optimization and stock management.
    v2: Added structured response hooks.
    """

    def __init__(self):
        super().__init__(name="Inventory Optimization Agent", model_name="gemini-2.5-flash")

    def get_system_prompt(self) -> str:
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
        """Process inventory optimization request (unchanged)"""
        logger.info(f"{self.name}: Processing inventory optimization request")

        forecasts = context.get("forecasts")
        if forecasts is not None:
            validate_dataframe(
                forecasts,
                ["store_id", "dept_id", "forecast_date", "predicted_sales"],
                "forecasts",
                date_columns=["forecast_date"],
            )

        question = context.get("question", "Provide inventory optimization recommendations based on the forecast.")

        opt_context = self._build_optimization_context(forecasts, context)
        response = self.generate_response(question, opt_context)

        return {
            "agent": self.name,
            "response": response,
            "timestamp": pd.Timestamp.now(),
            "context_summary": self._summarize_context(context),
        }

    # ------------------------------------------------------------------
    # v2: Structured response hooks
    # ------------------------------------------------------------------

    def _build_summary(self, context: Dict[str, Any], llm_response: str) -> str:
        """Build summary from actual inventory parameters and forecast data."""
        forecasts = context.get("forecasts")
        service_level = context.get("service_level", 0.95)
        lead_time = context.get("lead_time_days", 7)
        scope = self._summarize_context(context)

        if forecasts is None or len(forecasts) == 0:
            return f"Inventory optimization for {scope}: no forecast data available."

        total_demand = forecasts["predicted_sales"].sum()
        num_weeks = forecasts["forecast_date"].nunique()
        avg_weekly = forecasts["predicted_sales"].mean()
        demand_std = forecasts["predicted_sales"].std()

        # Simple safety stock estimate (z * std * sqrt(lead_time_in_weeks))
        from scipy.stats import norm
        z_score = norm.ppf(service_level)
        lead_time_weeks = lead_time / 7
        safety_stock_est = z_score * demand_std * np.sqrt(lead_time_weeks)

        return (
            f"[{scope}] Inventory plan for {num_weeks} weeks: avg weekly demand "
            f"\\${avg_weekly:,.2f}, total \\${total_demand:,.2f}. "
            f"At {service_level*100:.0f}% service level with {lead_time}-day lead time, "
            f"estimated safety stock buffer is \\${safety_stock_est:,.2f}/week."
        )

    def _extract_insights(self, context: Dict[str, Any], llm_response: str) -> List[Insight]:
        """
        Extract inventory-specific insights from forecast data.
        Computes safety stock, ROP, and risk metrics directly from data
        so they're structured and accessible — not buried in LLM prose.
        """
        insights = []
        forecasts = context.get("forecasts")

        if forecasts is None or len(forecasts) == 0:
            return insights

        service_level = context.get("service_level", 0.95)
        lead_time_days = context.get("lead_time_days", 7)

        avg_demand = forecasts["predicted_sales"].mean()
        demand_std = forecasts["predicted_sales"].std()
        cv = demand_std / avg_demand if avg_demand > 0 else 0

        # --- Insight 1: Computed safety stock & ROP (hard numbers) ---
        try:
            from scipy.stats import norm
            z = norm.ppf(service_level)
        except ImportError:
            # Fallback lookup if scipy not installed
            z_lookup = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
            z = z_lookup.get(service_level, 1.645)

        lead_time_weeks = lead_time_days / 7
        safety_stock = z * demand_std * np.sqrt(lead_time_weeks)
        daily_demand = avg_demand / 7
        rop = (daily_demand * lead_time_days) + safety_stock

        insights.append(Insight(
            title="Safety Stock Target",
            detail=(
                f"At {service_level*100:.0f}% service level with {lead_time_days}-day lead time: "
                f"safety stock = \\${safety_stock:,.2f}, "
                f"reorder point (ROP) = \\${rop:,.2f}. "
                f"Trigger reorder when total inventory drops to ROP."
            ),
            severity=InsightSeverity.INFO,
            metric_value=round(safety_stock, 2),
            metric_label="safety_stock_dollars",
        ))

        # --- Insight 2: Demand variability by store ---
        if "store_id" in forecasts.columns:
            store_cv = forecasts.groupby("store_id")["predicted_sales"].agg(
                lambda x: x.std() / x.mean() if x.mean() > 0 else 0
            )
            high_var_stores = store_cv[store_cv > 0.5]

            if len(high_var_stores) > 0:
                store_list = ", ".join(str(s) for s in high_var_stores.index[:5])
                insights.append(Insight(
                    title="High Demand Variability Stores",
                    detail=(
                        f"Stores {store_list} show high forecast variability (CV > 0.5). "
                        f"These need larger safety stock buffers to maintain "
                        f"{service_level*100:.0f}% service level."
                    ),
                    severity=InsightSeverity.WARNING,
                    metric_value=round(high_var_stores.max(), 2),
                    metric_label="max_store_cv",
                ))

        # --- Insight 3: Stockout risk assessment ---
        if cv > 0.7:
            insights.append(Insight(
                title="Elevated Stockout Risk",
                detail=(
                    f"Overall demand CV is {cv:.2f} — highly unpredictable. "
                    f"Standard safety stock (\\${safety_stock:,.2f}) may underestimate "
                    f"the buffer needed. Consider bumping service level to 97-99% "
                    f"for high-revenue departments."
                ),
                severity=InsightSeverity.CRITICAL,
                metric_value=round(cv, 2),
                metric_label="overall_demand_cv",
            ))
        elif cv > 0.4:
            insights.append(Insight(
                title="Moderate Demand Uncertainty",
                detail=(
                    f"Demand CV of {cv:.2f} suggests moderate variability. "
                    f"Current safety stock of \\${safety_stock:,.2f} should be adequate "
                    f"but monitor closely during peak seasons."
                ),
                severity=InsightSeverity.WARNING,
                metric_value=round(cv, 2),
                metric_label="overall_demand_cv",
            ))

        # --- Insight 4: Holding cost awareness ---
        # Rule of thumb: annual holding cost = 20-25% of inventory value
        annual_holding_cost_est = safety_stock * 0.25
        insights.append(Insight(
            title="Estimated Holding Cost Impact",
            detail=(
                f"Maintaining \\${safety_stock:,.2f} in safety stock has an estimated "
                f"annual holding cost of ~\\${annual_holding_cost_est:,.2f} "
                f"(assuming 25% annual carrying rate). "
                f"Weigh against stockout costs for your margin profile."
            ),
            severity=InsightSeverity.INFO,
            metric_value=round(annual_holding_cost_est, 2),
            metric_label="est_annual_holding_cost",
        ))

        return insights

    # ------------------------------------------------------------------
    # Existing methods (unchanged)
    # ------------------------------------------------------------------

    def _build_optimization_context(self, forecasts, context):
        """Build context for optimization"""
        analysis = {}

        if forecasts is not None and len(forecasts) > 0:
            weekly_demand = forecasts.groupby("store_id")["predicted_sales"].agg(["mean", "std", "sum"])

            analysis["demand_analysis"] = {
                "data_granularity": "WEEKLY (each predicted_sales value = 7 days of sales)",
                "avg_weekly_demand": f"${weekly_demand['mean'].mean():,.2f}",
                "demand_variability_std": f"${weekly_demand['std'].mean():,.2f}",
                "total_forecasted_demand": f"${weekly_demand['sum'].sum():,.2f}",
                "forecast_weeks": forecasts["forecast_date"].nunique(),
                "forecast_period": f"{forecasts['forecast_date'].min()} to {forecasts['forecast_date'].max()}",
                "num_stores": forecasts["store_id"].nunique(),
                "num_departments": forecasts["dept_id"].nunique(),
            }

            if "service_level" in context:
                analysis["service_level_target"] = f"{context['service_level']*100:.0f}%"
            if "lead_time_days" in context:
                analysis["lead_time"] = f"{context['lead_time_days']} days"
                analysis["lead_time_as_weeks"] = f"{context['lead_time_days'] / 7:.1f} weeks"

        return analysis

    def _summarize_context(self, context):
        summary = []
        if "store_id" in context and context["store_id"]:
            summary.append(f"Store {context['store_id']}")
        if "dept_id" in context and context["dept_id"]:
            summary.append(f"Department {context['dept_id']}")
        if "service_level" in context:
            summary.append(f"Service Level: {context['service_level']*100:.0f}%")
        return ", ".join(summary) if summary else "All stores and departments"

    def calculate_safety_stock(self, forecasts: pd.DataFrame, service_level: float = 0.95,
                               lead_time_days: int = 7) -> Dict[str, Any]:
        """Calculate safety stock — now returns AgentResponse via safe_process"""
        question = (
            f"Calculate and recommend safety stock levels for a {service_level*100:.0f}% service level "
            f"with {lead_time_days} days lead time. The demand data is WEEKLY. "
            f"Provide specific recommendations by store and department."
        )

        context = {
            "forecasts": forecasts,
            "service_level": service_level,
            "lead_time_days": lead_time_days,
            "question": question,
        }

        return self.safe_process(context)