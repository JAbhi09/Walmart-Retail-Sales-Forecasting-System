"""
Standardized Agent Response Model
Ensures all agents return structured, predictable results.

Works alongside the existing Gemini-based BaseAgent — does NOT replace it.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import pandas as pd


class AgentStatus(Enum):
    """Status of an agent's execution"""
    SUCCESS = "success"
    PARTIAL = "partial"       # Agent ran but with warnings (e.g., missing optional data)
    FAILURE = "failure"       # Agent crashed or LLM call failed
    SKIPPED = "skipped"       # Agent was not applicable for this query


class InsightSeverity(Enum):
    """Severity level for individual insights"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Insight:
    """
    A single structured insight extracted from agent analysis.
    
    Example:
        Insight(
            title="Demand Spike Detected",
            detail="Store 5, Dept 12 shows 40% above-average demand for weeks 48-51",
            severity=InsightSeverity.WARNING,
            metric_value=1.4,
            metric_label="demand_multiplier"
        )
    """
    title: str
    detail: str
    severity: InsightSeverity = InsightSeverity.INFO
    metric_value: Optional[float] = None
    metric_label: Optional[str] = None


@dataclass
class AgentResponse:
    """
    Standardized response that wraps every agent's output.
    
    The key improvement: each agent provides its own `summary` field,
    so the orchestrator never has to truncate or guess.
    
    Attributes:
        agent_name:       Which agent produced this (matches BaseAgent.name)
        status:           Execution outcome
        summary:          1-3 sentence human-readable summary (agents write this themselves)
        llm_response:     Full Gemini response text (the existing 'response' field)
        insights:         Structured actionable insights (optional, agent extracts these)
        recommendations:  Plain-text recommended actions
        raw_data:         Any computed DataFrames, metrics, or dicts
        metadata:         Execution metadata (timing, parameters, context_summary)
        error_message:    Error details if status is FAILURE or PARTIAL
    """
    agent_name: str
    status: AgentStatus
    summary: str
    llm_response: str = ""
    insights: List[Insight] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON/API/Streamlit responses"""
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "summary": self.summary,
            "llm_response": self.llm_response,
            "insights": [
                {
                    "title": i.title,
                    "detail": i.detail,
                    "severity": i.severity.value,
                    "metric_value": i.metric_value,
                    "metric_label": i.metric_label,
                }
                for i in self.insights
            ],
            "recommendations": self.recommendations,
            "metadata": self.metadata,
            "error_message": self.error_message,
            # raw_data excluded — may contain DataFrames or large objects
        }

    @property
    def has_critical_insights(self) -> bool:
        return any(i.severity == InsightSeverity.CRITICAL for i in self.insights)

    @property
    def insight_count(self) -> int:
        return len(self.insights)

