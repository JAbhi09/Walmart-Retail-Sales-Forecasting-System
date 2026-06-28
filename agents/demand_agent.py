"""
Demand Forecasting Agent (v2)
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


class DemandForecastingAgent(BaseAgent):
    """
    AI agent specialized in demand forecasting analysis and recommendations.
    v2: Added structured response hooks.
    """

    def __init__(self):
        super().__init__(name="Demand Forecasting Agent", model_name="gemini-2.5-flash")

    def get_system_prompt(self) -> str:
        return """You are an expert Demand Forecasting AI Agent for Walmart retail operations.

ABSOLUTE RULES (never violate these):
1. You must ONLY analyze data that is provided to you in the CONTEXT section.
2. If the user asks about a time period, category, or metric NOT present in the provided data, you MUST:
   - State clearly: "The available data does not cover [what they asked about]."
   - State what time period/scope the data DOES cover.
   - Offer to analyze what IS available.
   - STOP. Do NOT fill in with general retail knowledge, industry trends, or assumptions.
3. NEVER generate analysis from your training knowledge when specific data is required but unavailable.
4. Lead with the direct answer in 1-2 sentences before showing any supporting detail.
5. Keep responses concise and actionable — managers need decisions, not textbooks.

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
- Clear, data-driven insights with specific numbers from the data
- Specific, actionable recommendations
- Confidence levels for predictions
- Risk factors and mitigation strategies

Format your responses in a professional, concise manner suitable for retail managers."""

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process demand forecasting request"""
        logger.info(f"{self.name}: Processing demand forecasting request")

        forecasts = context.get("forecasts")
        if forecasts is not None:
            validate_dataframe(
                forecasts,
                ["store_id", "dept_id", "forecast_date", "predicted_sales"],
                "forecasts",
                date_columns=["forecast_date"],
            )

        historical_sales = context.get("historical_sales")
        question = context.get("question", "Analyze the demand forecast and provide insights.")

        # --- Pre-LLM data coverage gate ---
        # Check if the question asks about a time period we don't have data for
        coverage_block = self._check_data_coverage(question, forecasts, historical_sales)
        if coverage_block is not None:
            # Short-circuit: return the coverage message without calling Gemini
            logger.info(f"{self.name}: Question blocked by data coverage check")
            return {
                "agent": self.name,
                "response": coverage_block,
                "timestamp": pd.Timestamp.now(),
                "context_summary": self._summarize_context(context),
            }

        # --- Data grounding note (for questions that DO match our data) ---
        coverage_note = self._get_data_coverage_note(forecasts, historical_sales)
        grounded_question = f"{coverage_note}\n\nUser question: {question}"

        analysis_context = self._build_analysis_context(forecasts, historical_sales, context)
        response = self.generate_response(grounded_question, analysis_context)

        return {
            "agent": self.name,
            "response": response,
            "timestamp": pd.Timestamp.now(),
            "context_summary": self._summarize_context(context),
        }

    def _check_data_coverage(self, question: str, forecasts, historical_sales) -> str | None:
        """
        Pre-LLM gate: Check if the question asks about time periods
        not covered by the available data. Returns a response string
        if the question should be blocked, or None if it's fine to proceed.
        """
        question_lower = question.lower()

        # Determine what quarters/months the user is asking about
        quarter_keywords = {
            "q1": [1, 2, 3],
            "q2": [4, 5, 6],
            "q3": [7, 8, 9],
            "q4": [10, 11, 12],
        }
        month_keywords = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }

        # Collect all months present in our data
        available_months = set()
        if forecasts is not None and len(forecasts) > 0:
            available_months.update(pd.to_datetime(forecasts["forecast_date"]).dt.month.unique())
        if historical_sales is not None and len(historical_sales) > 0:
            available_months.update(pd.to_datetime(historical_sales["feature_date"]).dt.month.unique())

        if not available_months:
            return None  # No data at all — let Gemini handle the "no data" message

        # Check if user asked about a specific quarter
        asked_months = set()
        for q_key, q_months in quarter_keywords.items():
            if q_key in question_lower:
                asked_months.update(q_months)

        # Check if user asked about a specific month
        for month_name, month_num in month_keywords.items():
            if month_name in question_lower:
                asked_months.add(month_num)

        if not asked_months:
            return None  # No specific time period asked — proceed normally

        # Check overlap
        overlap = asked_months & available_months
        if overlap:
            return None  # Data covers at least part of what they asked — proceed

        # --- BLOCKED: No data for the requested period ---
        # Build a helpful response without calling Gemini
        asked_names = sorted(asked_months)
        available_names = sorted(available_months)

        # Format date ranges
        data_ranges = []
        if forecasts is not None and len(forecasts) > 0:
            f_min = forecasts["forecast_date"].min()
            f_max = forecasts["forecast_date"].max()
            data_ranges.append(f"Forecast data: {f_min} to {f_max}")
        if historical_sales is not None and len(historical_sales) > 0:
            h_min = historical_sales["feature_date"].min()
            h_max = historical_sales["feature_date"].max()
            data_ranges.append(f"Historical data: {h_min} to {h_max}")

        month_names = [
            "", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        asked_str = ", ".join(month_names[m] for m in sorted(asked_months))
        available_str = ", ".join(month_names[m] for m in sorted(available_months))

        return (
            f"### Data Coverage Notice\n\n"
            f"The available data **does not cover** the time period you asked about "
            f"({asked_str}).\n\n"
            f"**Available data covers:**\n"
            + "\n".join(f"- {r}" for r in data_ranges)
            + f"\n- Months with data: {available_str}\n\n"
            f"**What I can do instead:**\n"
            f"- Analyze demand trends within the available date range\n"
            f"- Compare historical patterns across the months we have data for\n"
            f"- Provide store/department-level insights for the covered period\n\n"
            f"Would you like me to analyze the data we do have?"
        )

    def _get_data_coverage_note(self, forecasts, historical_sales) -> str:
        """
        Build a note about what time periods the data actually covers.
        This prevents Gemini from hallucinating analysis for uncovered periods.
        """
        parts = ["DATA COVERAGE (only answer based on data within these ranges):"]

        if forecasts is not None and len(forecasts) > 0:
            f_min = forecasts["forecast_date"].min()
            f_max = forecasts["forecast_date"].max()
            parts.append(f"- Forecast data: {f_min} to {f_max}")
        else:
            parts.append("- Forecast data: NONE AVAILABLE")

        if historical_sales is not None and len(historical_sales) > 0:
            h_min = historical_sales["feature_date"].min()
            h_max = historical_sales["feature_date"].max()
            parts.append(f"- Historical data: {h_min} to {h_max}")
        else:
            parts.append("- Historical data: NONE AVAILABLE")

        parts.append(
            "IMPORTANT: If the user asks about a time period NOT covered by the data above, "
            "clearly state that the data does not cover that period. Do NOT generate analysis "
            "from general knowledge. Instead, describe what data IS available and offer to "
            "analyze that."
        )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # v2: Structured response hooks
    # ------------------------------------------------------------------

    def _build_summary(self, context: Dict[str, Any], llm_response: str) -> str:
        """
        Build summary from actual forecast data, not LLM truncation.
        """
        forecasts = context.get("forecasts")
        historical_sales = context.get("historical_sales")
        scope = self._summarize_context(context)

        parts = []

        if forecasts is not None and len(forecasts) > 0:
            avg_pred = forecasts["predicted_sales"].mean()
            num_weeks = forecasts["forecast_date"].nunique()
            total_pred = forecasts["predicted_sales"].sum()
            parts.append(
                f"Forecast covers {num_weeks} weeks with avg weekly predicted sales "
                f"of \\${avg_pred:,.2f} (total: \\${total_pred:,.2f})."
            )

        if historical_sales is not None and len(historical_sales) > 0:
            hist_avg = historical_sales["weekly_sales"].mean()
            if forecasts is not None and len(forecasts) > 0:
                change_pct = ((avg_pred - hist_avg) / hist_avg) * 100
                direction = "above" if change_pct > 0 else "below"
                parts.append(
                    f"Predicted demand is {abs(change_pct):.1f}% {direction} "
                    f"the historical weekly average of \\${hist_avg:,.2f}."
                )

        if not parts:
            return f"{self.name} analysis complete for {scope}."

        return f"[{scope}] " + " ".join(parts)

    def _extract_insights(self, context: Dict[str, Any], llm_response: str) -> List[Insight]:
        """Extract data-driven insights from forecast vs historical comparison."""
        insights = []
        forecasts = context.get("forecasts")
        historical_sales = context.get("historical_sales")

        if forecasts is None or historical_sales is None:
            return insights

        if len(forecasts) == 0 or len(historical_sales) == 0:
            return insights

        # Insight 1: Overall demand shift
        avg_pred = forecasts["predicted_sales"].mean()
        hist_avg = historical_sales["weekly_sales"].mean()
        change_pct = ((avg_pred - hist_avg) / hist_avg) * 100

        if abs(change_pct) > 20:
            severity = InsightSeverity.CRITICAL
        elif abs(change_pct) > 10:
            severity = InsightSeverity.WARNING
        else:
            severity = InsightSeverity.INFO

        direction = "increase" if change_pct > 0 else "decrease"
        insights.append(Insight(
            title=f"Demand {direction.title()} Detected",
            detail=(
                f"Forecasted average weekly sales (\\${avg_pred:,.2f}) represent a "
                f"{abs(change_pct):.1f}% {direction} from historical average (\\${hist_avg:,.2f})."
            ),
            severity=severity,
            metric_value=round(change_pct, 2),
            metric_label="demand_change_pct",
        ))

        # Insight 2: High-variance departments
        if "dept_id" in forecasts.columns:
            dept_stats = forecasts.groupby("dept_id")["predicted_sales"].agg(["mean", "std"])
            dept_stats["cv"] = dept_stats["std"] / dept_stats["mean"]  # coefficient of variation
            volatile_depts = dept_stats[dept_stats["cv"] > 0.5]

            if len(volatile_depts) > 0:
                dept_list = ", ".join(str(d) for d in volatile_depts.index[:5])
                insights.append(Insight(
                    title="High-Variance Departments",
                    detail=(
                        f"Departments {dept_list} show high forecast variability "
                        f"(CV > 0.5), indicating uncertain demand patterns."
                    ),
                    severity=InsightSeverity.WARNING,
                    metric_value=round(volatile_depts["cv"].max(), 2),
                    metric_label="max_coefficient_of_variation",
                ))

        # Insight 3: Recent trend from historical data
        if "feature_date" in historical_sales.columns:
            sorted_hist = historical_sales.sort_values("feature_date", ascending=False)
            dates = sorted_hist["feature_date"].unique()
            if len(dates) >= 16:
                recent = sorted_hist[sorted_hist["feature_date"].isin(dates[:8])]["weekly_sales"].mean()
                prior = sorted_hist[sorted_hist["feature_date"].isin(dates[8:16])]["weekly_sales"].mean()
                trend_pct = ((recent - prior) / prior) * 100

                if abs(trend_pct) > 15:
                    trend_dir = "upward" if trend_pct > 0 else "downward"
                    insights.append(Insight(
                        title=f"Strong {trend_dir.title()} Trend",
                        detail=(
                            f"Recent 8 weeks avg (\\${recent:,.2f}) vs prior 8 weeks "
                            f"(\\${prior:,.2f}) shows a {abs(trend_pct):.1f}% {trend_dir} shift."
                        ),
                        severity=InsightSeverity.WARNING,
                        metric_value=round(trend_pct, 2),
                        metric_label="trend_change_pct",
                    ))

        return insights

    # ------------------------------------------------------------------
    # Existing methods (unchanged)
    # ------------------------------------------------------------------

    def _build_analysis_context(self, forecasts, historical_sales, context):
        """Build context for analysis"""
        analysis = {}

        if forecasts is not None and len(forecasts) > 0:
            analysis["forecast_summary"] = {
                "data_granularity": "WEEKLY (each value = one full week of sales)",
                "total_predicted_weekly_sales": f"${forecasts['predicted_sales'].sum():,.2f}",
                "avg_predicted_weekly_sales": f"${forecasts['predicted_sales'].mean():,.2f}",
                "min_predicted_weekly_sales": f"${forecasts['predicted_sales'].min():,.2f}",
                "max_predicted_weekly_sales": f"${forecasts['predicted_sales'].max():,.2f}",
                "forecast_period": f"{forecasts['forecast_date'].min()} to {forecasts['forecast_date'].max()}",
                "num_weeks_forecasted": forecasts["forecast_date"].nunique(),
                "num_store_dept_combinations": len(forecasts.groupby(["store_id", "dept_id"])),
            }

            if forecasts["store_id"].nunique() <= 10:
                store_summary = forecasts.groupby("store_id")["predicted_sales"].agg(["mean", "sum"]).round(2)
                analysis["forecast_by_store"] = store_summary.to_dict()

        if historical_sales is not None and len(historical_sales) > 0:
            analysis["historical_summary"] = {
                "data_granularity": "WEEKLY (each value = one full week of sales)",
                "avg_weekly_sales": f"${historical_sales['weekly_sales'].mean():,.2f}",
                "median_weekly_sales": f"${historical_sales['weekly_sales'].median():,.2f}",
                "std_weekly_sales": f"${historical_sales['weekly_sales'].std():,.2f}",
                "total_historical_sales": f"${historical_sales['weekly_sales'].sum():,.2f}",
                "historical_date_range": f"{historical_sales['feature_date'].min()} to {historical_sales['feature_date'].max()}",
                "num_historical_weeks": historical_sales["feature_date"].nunique(),
            }

            sorted_hist = historical_sales.sort_values("feature_date", ascending=False)
            dates = sorted_hist["feature_date"].unique()
            if len(dates) >= 16:
                recent_avg = sorted_hist[sorted_hist["feature_date"].isin(dates[:8])]["weekly_sales"].mean()
                prior_avg = sorted_hist[sorted_hist["feature_date"].isin(dates[8:16])]["weekly_sales"].mean()
                trend_pct = ((recent_avg - prior_avg) / prior_avg) * 100
                analysis["recent_trend"] = {
                    "recent_8wk_avg": f"${recent_avg:,.2f}",
                    "prior_8wk_avg": f"${prior_avg:,.2f}",
                    "trend_change_pct": f"{trend_pct:+.1f}%",
                }

        if "store_id" in context and context["store_id"]:
            analysis["scope"] = f"Store {context['store_id']}"
        if "dept_id" in context and context["dept_id"]:
            analysis["scope"] = analysis.get("scope", "") + f", Department {context['dept_id']}"

        return analysis

    def _summarize_context(self, context):
        summary = []
        if "store_id" in context and context["store_id"]:
            summary.append(f"Store {context['store_id']}")
        if "dept_id" in context and context["dept_id"]:
            summary.append(f"Department {context['dept_id']}")
        return ", ".join(summary) if summary else "All stores and departments"

    def analyze_forecast_accuracy(self, actual: pd.DataFrame, predicted: pd.DataFrame) -> Dict[str, Any]:
        """Analyze forecast accuracy — now returns AgentResponse via safe_process"""
        question = """Analyze the forecast accuracy comparing actual vs predicted weekly sales.
        Identify where the model performs well and where it struggles.
        Provide specific recommendations to improve forecast accuracy."""

        merged = actual.merge(predicted, on=["store_id", "dept_id", "feature_date"], how="inner")
        mae = np.abs(merged["weekly_sales"] - merged["predicted_sales"]).mean()
        mape = (np.abs(merged["weekly_sales"] - merged["predicted_sales"]) / merged["weekly_sales"]).mean() * 100

        context = {
            "actual_sales": actual,
            "predicted_sales": predicted,
            "accuracy_metrics": {"mae": f"${mae:,.2f}", "mape": f"{mape:.2f}%", "num_comparisons": len(merged)},
            "question": question,
        }

        return self.safe_process(context)