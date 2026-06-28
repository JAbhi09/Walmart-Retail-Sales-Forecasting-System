"""
Anomaly Detection Agent (v2)
Overrides summary/insight hooks for structured responses.
All existing functionality preserved — only added hook overrides.
"""
import yaml
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, Any, List
import logging

from agents.base_agent import BaseAgent
from agents.response_model import Insight, InsightSeverity
from agents.validation import validate_dataframe

logger = logging.getLogger(__name__)


class AnomalyDetectionAgent(BaseAgent):
    """
    AI agent specialized in detecting anomalies and unusual patterns.
    
    v2: Added _build_summary, _extract_insights, _extract_recommendations
    """

    def __init__(self):
        super().__init__(name="Anomaly Detection Agent", model_name="gemini-2.5-flash")
        self.threshold = self._load_threshold()
        logger.info(f"{self.name}: anomaly detection threshold = {self.threshold}")

    @staticmethod
    def _load_threshold() -> float:
        """Load z-score threshold from config/config.yaml, falling back to 3.0."""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            return float(cfg.get("anomaly_detection", {}).get("threshold", 3.0))
        except Exception:
            return 3.0

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
        Process anomaly detection request (unchanged from v1).
        Returns raw dict for backward compatibility.
        safe_process() wraps this with AgentResponse.
        """
        logger.info(f"{self.name}: Processing anomaly detection request")

        sales_data = context.get("sales_data")
        if sales_data is not None:
            validate_dataframe(
                sales_data,
                ["store_id", "dept_id", "feature_date", "weekly_sales"],
                "sales_data",
                date_columns=["feature_date"],
            )

        anomalies = context.get("anomalies", [])
        question = context.get("question", "Analyze the detected anomalies and provide insights.")

        anomaly_context = self._build_anomaly_context(sales_data, anomalies, context)
        response = self.generate_response(question, anomaly_context)

        return {
            "agent": self.name,
            "response": response,
            "timestamp": pd.Timestamp.now(),
            "anomalies_detected": len(anomalies),
            "context_summary": self._summarize_context(context),
        }

    # ------------------------------------------------------------------
    # v2: Structured response hooks
    # ------------------------------------------------------------------

    def _build_summary(self, context: Dict[str, Any], llm_response: str) -> str:
        """
        Build a concise summary using actual anomaly data, not truncation.
        """
        anomalies = context.get("anomalies", [])
        sales_data = context.get("sales_data")
        threshold = context.get("threshold", "N/A")

        count = len(anomalies)
        if count == 0:
            return (
                f"No anomalies detected at threshold {threshold}. "
                "Sales data appears within normal statistical bounds."
            )

        # Determine severity from z-scores
        z_scores = []
        for a in anomalies:
            try:
                z_scores.append(float(a.get("z_score", 0)))
            except (ValueError, TypeError):
                pass

        max_z = max(z_scores) if z_scores else 0
        severity = "critical" if max_z > 5 else "high" if max_z > 4 else "moderate"

        scope = self._summarize_context(context)
        summary = (
            f"Detected {count} anomalies ({severity} severity, max z-score: {max_z:.1f}) "
            f"across {scope} using threshold {threshold}."
        )

        # Add top anomaly detail if available
        if anomalies:
            top = anomalies[0]
            summary += (
                f" Most significant: Store {top.get('store_id')}, "
                f"Dept {top.get('dept_id')} on {top.get('date')} "
                f"with sales of {top.get('sales')}."
            )

        return summary

    def _extract_insights(self, context: Dict[str, Any], llm_response: str) -> List[Insight]:
        """
        Extract structured insights from anomaly detection results.
        Uses the actual anomaly data for precision rather than parsing LLM text.
        """
        anomalies = context.get("anomalies", [])
        insights = []

        for anomaly in anomalies[:10]:  # Top 10
            try:
                z_score = float(anomaly.get("z_score", 0))
            except (ValueError, TypeError):
                z_score = 0

            # Map z-score to severity
            if z_score > 5:
                severity = InsightSeverity.CRITICAL
            elif z_score > 4:
                severity = InsightSeverity.WARNING
            else:
                severity = InsightSeverity.INFO

            insights.append(Insight(
                title=f"Anomaly: Store {anomaly.get('store_id')}, Dept {anomaly.get('dept_id')}",
                detail=(
                    f"On {anomaly.get('date')}, weekly sales of {anomaly.get('sales')} "
                    f"deviated by {anomaly.get('deviation', 'N/A')} from the mean "
                    f"(z-score: {anomaly.get('z_score')})."
                ),
                severity=severity,
                metric_value=z_score,
                metric_label="z_score",
            ))

        return insights

    def _extract_recommendations(self, context: Dict[str, Any], llm_response: str) -> List[str]:
        """
        Combine data-driven recommendations with any LLM-parsed ones.
        """
        anomalies = context.get("anomalies", [])
        recs = []

        # Data-driven recommendations
        if len(anomalies) > 10:
            recs.append(
                f"Investigate the {len(anomalies)} anomalies systematically — "
                "prioritize those with z-scores above 4.0 for immediate review."
            )

        critical = [a for a in anomalies if float(a.get("z_score", 0)) > 5]
        if critical:
            stores = set(a.get("store_id") for a in critical)
            recs.append(
                f"Critical anomalies in store(s) {stores}: verify data quality first, "
                "then check for promotional events or recording errors."
            )

        # Also pull LLM-parsed recommendations from parent
        llm_recs = super()._extract_recommendations(context, llm_response)
        recs.extend(llm_recs)

        return recs[:5]

    # ------------------------------------------------------------------
    # Existing methods (unchanged)
    # ------------------------------------------------------------------

    def _build_anomaly_context(self, sales_data, anomalies, context):
        """Build context for anomaly analysis"""
        analysis = {}

        if sales_data is not None and len(sales_data) > 0:
            analysis["data_summary"] = {
                "total_records": len(sales_data),
                "date_range": f"{sales_data['feature_date'].min()} to {sales_data['feature_date'].max()}",
                "avg_sales": f"${sales_data['weekly_sales'].mean():,.2f}",
                "sales_std": f"${sales_data['weekly_sales'].std():,.2f}",
            }

        if anomalies:
            analysis["anomalies_found"] = len(anomalies)
            analysis["anomaly_details"] = anomalies[:10]

        if "threshold" in context:
            analysis["detection_threshold"] = context["threshold"]

        return analysis

    def _summarize_context(self, context):
        """Create a summary of the context"""
        summary = []
        if "store_id" in context:
            summary.append(f"Store {context['store_id']}")
        if "dept_id" in context:
            summary.append(f"Department {context['dept_id']}")
        return ", ".join(summary) if summary else "All stores and departments"

    def detect_anomalies(self, sales_data: pd.DataFrame, threshold: float = None) -> Dict[str, Any]:
        """
        Detect anomalies using statistical methods.
        Now returns AgentResponse via safe_process() instead of raw dict.
        """
        validate_dataframe(
            sales_data,
            ["store_id", "dept_id", "feature_date", "weekly_sales"],
            "sales_data",
            date_columns=["feature_date"],
        )

        if threshold is None:
            threshold = self.threshold
        logger.info(f"{self.name}: Detecting anomalies with threshold={threshold}")

        sales_data = sales_data.copy()
        sales_data["z_score"] = np.abs(
            (sales_data["weekly_sales"] - sales_data["weekly_sales"].mean())
            / sales_data["weekly_sales"].std()
        )

        anomalies_df = sales_data[sales_data["z_score"] > threshold]

        anomalies = []
        for _, row in anomalies_df.head(20).iterrows():
            anomalies.append({
                "date": str(row["feature_date"]),
                "store_id": int(row["store_id"]),
                "dept_id": int(row["dept_id"]),
                "sales": f"${row['weekly_sales']:,.2f}",
                "z_score": f"{row['z_score']:.2f}",
                "deviation": f"{(row['z_score'] * sales_data['weekly_sales'].std()):,.2f}",
            })

        question = (
            f"Analyze these {len(anomalies_df)} detected anomalies. "
            "Identify patterns, assess severity, and provide recommendations."
        )

        context = {
            "sales_data": sales_data,
            "anomalies": anomalies,
            "threshold": threshold,
            "question": question,
        }

        # v2: Use safe_process for structured response
        return self.safe_process(context)