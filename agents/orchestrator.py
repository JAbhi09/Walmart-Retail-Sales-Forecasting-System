"""
Multi-Agent Orchestrator
Coordinates multiple AI agents to provide comprehensive insights
"""
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

from agents.demand_agent import DemandForecastingAgent
from agents.inventory_agent import InventoryOptimizationAgent
from agents.anomaly_agent import AnomalyDetectionAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestrates multiple AI agents to provide comprehensive analysis
    """
    
    def __init__(self):
        """Initialize orchestrator with all agents"""
        self.agents = {
            'demand': DemandForecastingAgent(),
            'inventory': InventoryOptimizationAgent(),
            'anomaly': AnomalyDetectionAgent()
        }
        
        logger.info("✓ AgentOrchestrator initialized with 3 agents")
    
    def analyze_forecast(self, forecasts: pd.DataFrame, historical_sales: pd.DataFrame,
                        store_id: Optional[int] = None, dept_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Comprehensive forecast analysis using multiple agents
        
        Args:
            forecasts: DataFrame with predictions
            historical_sales: DataFrame with historical data
            store_id: Optional store filter
            dept_id: Optional department filter
        
        Returns:
            Dict with insights from all agents
        """
        logger.info("AgentOrchestrator: Running comprehensive forecast analysis")
        
        # Filter data if needed
        if store_id:
            forecasts = forecasts[forecasts['store_id'] == store_id]
            historical_sales = historical_sales[historical_sales['store_id'] == store_id]
        if dept_id:
            forecasts = forecasts[forecasts['dept_id'] == dept_id]
            historical_sales = historical_sales[historical_sales['dept_id'] == dept_id]
        
        results = {}
        
        # 1. Demand Forecasting Analysis
        logger.info("  Running Demand Forecasting Agent...")
        demand_context = {
            'forecasts': forecasts,
            'historical_sales': historical_sales,
            'store_id': store_id,
            'dept_id': dept_id,
            'question': 'Analyze the sales forecast and provide insights on demand trends, patterns, and recommendations.'
        }
        results['demand_analysis'] = self.agents['demand'].process(demand_context)
        
        # 2. Inventory Optimization
        logger.info("  Running Inventory Optimization Agent...")
        inventory_context = {
            'forecasts': forecasts,
            'service_level': 0.95,
            'lead_time_days': 7,
            'store_id': store_id,
            'dept_id': dept_id,
            'question': 'Provide inventory optimization recommendations based on the forecast.'
        }
        results['inventory_recommendations'] = self.agents['inventory'].process(inventory_context)
        
        # 3. Anomaly Detection
        logger.info("  Running Anomaly Detection Agent...")
        anomaly_result = self.agents['anomaly'].detect_anomalies(historical_sales, threshold=3.0)
        results['anomaly_detection'] = anomaly_result
        
        logger.info("✓ Comprehensive analysis complete")
        
        return {
            'summary': self._create_summary(results),
            'detailed_insights': results,
            'timestamp': pd.Timestamp.now()
        }
    
    def ask_agent(self, agent_name: str, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask a specific agent a question
        
        Args:
            agent_name: Name of agent ('demand', 'inventory', or 'anomaly')
            question: Question to ask
            context: Context data for the agent
        
        Returns:
            Dict with agent response
        """
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}. Available: {list(self.agents.keys())}")
        
        logger.info(f"AgentOrchestrator: Asking {agent_name} agent")
        
        context['question'] = question
        return self.agents[agent_name].process(context)
    
    def get_agent(self, agent_name: str):
        """
        Get a specific agent
        
        Args:
            agent_name: Name of agent
        
        Returns:
            Agent instance
        """
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}. Available: {list(self.agents.keys())}")
        
        return self.agents[agent_name]
    
    def _create_summary(self, results: Dict[str, Any]) -> str:
        """
        Create a summary of all agent insights
        
        Args:
            results: Results from all agents
        
        Returns:
            str: Summary text
        """
        summary_parts = []
        
        summary_parts.append("=== MULTI-AGENT ANALYSIS SUMMARY ===\n")
        
        if 'demand_analysis' in results:
            summary_parts.append("[DEMAND FORECASTING]:")
            summary_parts.append(f"  {results['demand_analysis']['response'][:200]}...\n")
        
        if 'inventory_recommendations' in results:
            summary_parts.append("[INVENTORY OPTIMIZATION]:")
            summary_parts.append(f"  {results['inventory_recommendations']['response'][:200]}...\n")
        
        if 'anomaly_detection' in results:
            anomalies_count = results['anomaly_detection'].get('anomalies_detected', 0)
            summary_parts.append(f"[ANOMALY DETECTION]:")
            summary_parts.append(f"  Detected {anomalies_count} anomalies\n")
        
        return "\n".join(summary_parts)
