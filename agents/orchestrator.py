"""
Multi-Agent Orchestrator (v2)
- Uses safe_process() → AgentResponse instead of raw dicts
- Per-agent error isolation (one failure doesn't kill the pipeline)
- Summary built from agent-provided summaries, not truncation
- Fixed the store_id=0 falsy bug
- Critical alerts roll-up across all agents
"""
import pandas as pd
from typing import Any, Dict, List, Optional
import logging

from agents.response_model import AgentResponse, AgentStatus, InsightSeverity
from agents.demand_agent import DemandForecastingAgent
from agents.inventory_agent import InventoryOptimizationAgent
from agents.anomaly_agent import AnomalyDetectionAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestrates multiple AI agents for comprehensive analysis.
    
    v2 Changes from v1:
        - Calls safe_process() instead of process() → structured AgentResponse
        - Each agent's own summary used (no more 200-char truncation)
        - Per-agent error isolation built into safe_process()
        - Fixed: store_id=0 no longer skips filtering
        - DataFrames are copied before filtering (no mutation side effects)
        - New: critical_alerts roll-up for dashboards/notifications
        - Backward compatible: agent_results dict still works for old code
    """

    def __init__(self):
        self.agents = {
            "demand": DemandForecastingAgent(),
            "inventory": InventoryOptimizationAgent(),
            "anomaly": AnomalyDetectionAgent(),
        }
        logger.info(f"✓ AgentOrchestrator initialized with {len(self.agents)} agents")

    def analyze_forecast(
        self,
        forecasts: pd.DataFrame,
        historical_sales: pd.DataFrame,
        store_id: Optional[int] = None,
        dept_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive forecast analysis using multiple agents.
        Each agent runs independently — if one fails, others still return.
        """
        logger.info("AgentOrchestrator: Running comprehensive forecast analysis")

        # --- Filter data (FIXED: 'is not None' instead of truthy check) ---
        filtered_forecasts = forecasts.copy()
        filtered_historical = historical_sales.copy()

        if store_id is not None:
            filtered_forecasts = filtered_forecasts[filtered_forecasts["store_id"] == store_id]
            filtered_historical = filtered_historical[filtered_historical["store_id"] == store_id]
        if dept_id is not None:
            filtered_forecasts = filtered_forecasts[filtered_forecasts["dept_id"] == dept_id]
            filtered_historical = filtered_historical[filtered_historical["dept_id"] == dept_id]

        # --- Run each agent via safe_process (errors isolated per agent) ---
        results: Dict[str, AgentResponse] = {}

        # 1. Demand Forecasting
        logger.info("  Running Demand Forecasting Agent...")
        results["demand"] = self.agents["demand"].safe_process({
            "forecasts": filtered_forecasts,
            "historical_sales": filtered_historical,
            "store_id": store_id,
            "dept_id": dept_id,
            "question": "Analyze the sales forecast and provide insights on demand trends, patterns, and recommendations.",
        })

        # 2. Inventory Optimization
        logger.info("  Running Inventory Optimization Agent...")
        results["inventory"] = self.agents["inventory"].safe_process({
            "forecasts": filtered_forecasts,
            "service_level": 0.95,
            "lead_time_days": 7,
            "store_id": store_id,
            "dept_id": dept_id,
            "question": "Provide inventory optimization recommendations based on the forecast.",
        })

        # 3. Anomaly Detection
        logger.info("  Running Anomaly Detection Agent...")
        results["anomaly"] = self.agents["anomaly"].safe_process({
            "sales_data": filtered_historical,
            "anomalies": [],  # Will be populated by agent's own detection
            "threshold": self.agents["anomaly"].threshold,
            "question": "Detect and analyze anomalies in historical sales data.",
        })

        logger.info("✓ Comprehensive analysis complete")

        # Cross-agent synthesis: find contradictions and connections
        cross_insights = self._synthesize_across_agents(results)

        return {
            "summary": self._create_summary(results),
            "cross_agent_synthesis": cross_insights,
            "critical_alerts": self._extract_critical_alerts(results),
            "agent_results": {name: resp.to_dict() for name, resp in results.items()},
            "raw_responses": results,
            "timestamp": pd.Timestamp.now(),
        }

    def ask_agent(self, agent_name: str, question: str, context: Dict[str, Any]):
        """Ask a specific agent a question. Returns AgentResponse from safe_process()."""
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}. Available: {list(self.agents.keys())}")

        logger.info(f"AgentOrchestrator: Asking {agent_name} agent")
        context["question"] = question
        return self.agents[agent_name].safe_process(context)

    def get_agent(self, agent_name: str):
        """Get a specific agent instance"""
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}. Available: {list(self.agents.keys())}")
        return self.agents[agent_name]

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _create_summary(self, results: Dict[str, AgentResponse]) -> str:
        """
        Build summary from each agent's own .summary field.
        No truncation — agents are responsible for their own summaries.
        """
        lines = ["=== MULTI-AGENT ANALYSIS SUMMARY ===\n"]

        agent_labels = {
            "demand": "DEMAND FORECASTING",
            "inventory": "INVENTORY OPTIMIZATION",
            "anomaly": "ANOMALY DETECTION",
        }

        for name, response in results.items():
            label = agent_labels.get(name, name.upper())
            icon = self._status_icon(response.status)

            lines.append(f"[{label}] {icon}")
            lines.append(f"  {response.summary}")

            if response.recommendations:
                lines.append(f"  → Top recommendation: {response.recommendations[0]}")

            if response.has_critical_insights:
                critical_count = sum(
                    1 for i in response.insights if i.severity == InsightSeverity.CRITICAL
                )
                lines.append(f"  ⚠ {critical_count} critical insight(s) — see details")

            lines.append("")

        # Overall health
        failed = [n for n, r in results.items() if r.status == AgentStatus.FAILURE]
        if failed:
            lines.append(f"⚠ DEGRADED: {', '.join(failed)} agent(s) failed. See error details.")
        else:
            lines.append("✓ All agents completed successfully.")

        return "\n".join(lines)

    def _extract_critical_alerts(self, results: Dict[str, AgentResponse]) -> List[Dict[str, Any]]:
        """
        Roll up all CRITICAL severity insights across agents.
        Useful for dashboards, Slack notifications, or alert systems.
        """
        alerts = []
        for name, response in results.items():
            for insight in response.insights:
                if insight.severity == InsightSeverity.CRITICAL:
                    alerts.append({
                        "agent": name,
                        "title": insight.title,
                        "detail": insight.detail,
                        "metric_value": insight.metric_value,
                        "metric_label": insight.metric_label,
                    })
        return alerts

    @staticmethod
    def _status_icon(status: AgentStatus) -> str:
        return {
            AgentStatus.SUCCESS: "✓",
            AgentStatus.PARTIAL: "⚠",
            AgentStatus.FAILURE: "✗",
            AgentStatus.SKIPPED: "○",
        }.get(status, "?")

    def _synthesize_across_agents(self, results: Dict[str, AgentResponse]) -> List[Dict[str, Any]]:
        """
        Cross-agent synthesis: find connections, contradictions,
        and combined insights that no single agent would catch alone.
        
        This is what makes the comprehensive view genuinely more valuable
        than running agents individually.
        """
        synthesis = []

        demand = results.get("demand")
        inventory = results.get("inventory")
        anomaly = results.get("anomaly")

        # --- Insight 1: Demand decline + inventory levels mismatch ---
        if demand and inventory:
            demand_change = None
            for insight in demand.insights:
                if insight.metric_label == "demand_change_pct":
                    demand_change = insight.metric_value
                    break

            safety_stock = None
            for insight in inventory.insights:
                if insight.metric_label == "safety_stock_dollars":
                    safety_stock = insight.metric_value
                    break

            if demand_change is not None and demand_change < -5 and safety_stock:
                synthesis.append({
                    "type": "contradiction",
                    "title": "Demand Decline vs Safety Stock Levels",
                    "detail": (
                        f"Demand forecast shows a {abs(demand_change):.1f}% decline, "
                        f"but safety stock (${safety_stock:,.2f}) is calculated from "
                        f"historical averages that include higher-demand periods. "
                        f"Consider recalculating safety stock using forecasted demand "
                        f"to avoid overstocking during a softer period."
                    ),
                    "agents": ["demand", "inventory"],
                    "action": "Recalculate safety stock using forecasted demand instead of historical average.",
                })

        # --- Insight 2: Anomalies affecting forecast reliability ---
        if anomaly and demand:
            critical_anomalies = [
                i for i in anomaly.insights
                if i.severity == InsightSeverity.CRITICAL
            ]
            if len(critical_anomalies) >= 3:
                synthesis.append({
                    "type": "dependency",
                    "title": "Anomalies May Be Skewing Forecasts",
                    "detail": (
                        f"{len(critical_anomalies)} critical anomalies detected in "
                        f"historical data. If these are data errors (not genuine sales), "
                        f"they inflate the historical average and demand variability, "
                        f"which directly affects both the forecast comparison and "
                        f"safety stock calculations."
                    ),
                    "agents": ["anomaly", "demand", "inventory"],
                    "action": (
                        "Resolve anomaly root causes first. If data errors are confirmed, "
                        "clean the data and re-run both forecast and inventory models."
                    ),
                })

        # --- Insight 3: High variability + high anomaly count ---
        if inventory and anomaly:
            high_cv = None
            for insight in inventory.insights:
                if insight.metric_label == "overall_demand_cv":
                    high_cv = insight.metric_value
                    break

            anomaly_count = len(anomaly.insights)
            if high_cv and high_cv > 0.7 and anomaly_count > 5:
                synthesis.append({
                    "type": "reinforcement",
                    "title": "Demand Volatility Confirmed by Both Agents",
                    "detail": (
                        f"Inventory agent flagged demand CV of {high_cv:.2f} (highly volatile), "
                        f"and anomaly agent found {anomaly_count} anomalies. These reinforce "
                        f"each other: the data genuinely has extreme variability. "
                        f"Standard safety stock formulas may not be sufficient."
                    ),
                    "agents": ["inventory", "anomaly"],
                    "action": (
                        "Consider tiered service levels: 97-99% for critical departments, "
                        "90% for low-impact ones. Also investigate if the volatility is "
                        "driven by a few departments that could be managed separately."
                    ),
                })

        # --- Insight 4: Anomalies concentrated on last date + forecast period starts right after ---
        if anomaly and demand:
            # Check if anomalies are on the boundary between historical and forecast
            last_date_anomalies = set()
            for insight in anomaly.insights:
                if "2012-10-26" in insight.detail:
                    last_date_anomalies.add(insight.title)

            if len(last_date_anomalies) >= 2:
                synthesis.append({
                    "type": "warning",
                    "title": "Anomalies at Data Boundary",
                    "detail": (
                        f"Multiple anomalies cluster on 2012-10-26, which is the last week "
                        f"of historical data — right before the forecast period begins. "
                        f"If these are data errors, the forecast model may have trained on "
                        f"corrupted end-of-period data, potentially biasing predictions."
                    ),
                    "agents": ["anomaly", "demand"],
                    "action": (
                        "Verify the 2012-10-26 data integrity before trusting the forecast. "
                        "If errors are found, retrain the model excluding that week."
                    ),
                })

        return synthesis