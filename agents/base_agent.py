"""
Base Agent Class for Multi-Agent AI System (v2)
Adds structured AgentResponse support while keeping all existing Gemini functionality.

Changes from v1:
  - Added _build_summary() hook for agents to provide their own summaries
  - Added _extract_insights() hook for structured insight extraction
  - Added _extract_recommendations() hook
  - Added safe_process() wrapper with timing + error isolation
  - All existing methods (generate_response, get_system_prompt, etc.) unchanged
"""
import os
import re
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

import google.generativeai as genai

from agents.response_model import (
    AgentResponse, AgentStatus, Insight, InsightSeverity
)

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for AI agents.
    
    Subclasses MUST implement:
        - get_system_prompt()
        - process(context)          # existing — returns raw dict (kept for compat)
    
    Subclasses SHOULD implement (for structured responses):
        - _build_summary(context, llm_response)
        - _extract_insights(context, llm_response)
        - _extract_recommendations(context, llm_response)
    """

    def __init__(self, name: str, model_name: str = "gemini-2.5-flash"):
        self.name = name
        self.model_name = model_name

        # Configure Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        # Agent state
        self.conversation_history = []

        logger.info(f"✓ {self.name} initialized with {model_name}")

    # ------------------------------------------------------------------
    # Abstract methods (unchanged)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request — returns raw dict.
        Kept for backward compatibility. Existing code calls this directly.
        """
        pass

    # ------------------------------------------------------------------
    # NEW: Structured response wrapper
    # ------------------------------------------------------------------

    def safe_process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Wrapper around process() that returns a structured AgentResponse.
        
        - Calls the existing process() method
        - Wraps the result with summary, insights, recommendations
        - Catches errors so one agent failure doesn't crash the pipeline
        - Adds execution timing metadata
        
        The orchestrator should call safe_process() instead of process().
        """
        start_time = time.time()
        logger.info(f"[{self.name}] Starting safe_process...")

        try:
            # Call the existing process() — returns raw dict
            raw_result = self.process(context)
            elapsed = round(time.time() - start_time, 3)

            llm_response = raw_result.get("response", "")

            # Build structured response using hooks
            summary = self._build_summary(context, llm_response)
            insights = self._extract_insights(context, llm_response)
            recommendations = self._extract_recommendations(context, llm_response)

            return AgentResponse(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                summary=summary,
                llm_response=llm_response,
                insights=insights,
                recommendations=recommendations,
                raw_data={
                    k: v for k, v in raw_result.items()
                    if k not in ("agent", "response", "timestamp")
                },
                metadata={
                    "execution_time_seconds": elapsed,
                    "timestamp": raw_result.get("timestamp", datetime.now()),
                    "context_summary": raw_result.get("context_summary", ""),
                    "model": self.model_name,
                },
            )

        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            logger.error(f"[{self.name}] Failed after {elapsed}s: {e}", exc_info=True)

            return AgentResponse(
                agent_name=self.name,
                status=AgentStatus.FAILURE,
                summary=f"{self.name} encountered an error and could not complete analysis.",
                error_message=str(e),
                metadata={
                    "execution_time_seconds": elapsed,
                    "model": self.model_name,
                },
            )

    # ------------------------------------------------------------------
    # Hooks for subclasses to override (with sensible defaults)
    # ------------------------------------------------------------------

    def _build_summary(self, context: Dict[str, Any], llm_response: str) -> str:
        """
        Build a 1-3 sentence summary of the agent's findings.
        
        Default: extracts the first 2 sentences from the LLM response.
        Override in subclasses for smarter summaries.
        """
        if not llm_response:
            return f"{self.name} produced no output."

        # Strip markdown headers and clean up
        clean = re.sub(r"^#{1,4}\s+.*$", "", llm_response, flags=re.MULTILINE).strip()
        # Split into sentences (simple heuristic)
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        # Take first 2 non-empty sentences
        meaningful = [s.strip() for s in sentences if len(s.strip()) > 20][:2]

        return " ".join(meaningful) if meaningful else clean[:300]

    def _extract_insights(self, context: Dict[str, Any], llm_response: str) -> List[Insight]:
        """
        Extract structured insights from the LLM response.
        
        Default: returns empty list.
        Override in subclasses to parse agent-specific patterns.
        """
        return []

    def _extract_recommendations(self, context: Dict[str, Any], llm_response: str) -> List[str]:
        """
        Extract recommendation strings from the LLM response.
        
        Default: looks for lines starting with bullet points or numbers
        under sections containing "recommend" or "action".
        Override for better extraction.
        """
        if not llm_response:
            return []

        recommendations = []
        in_rec_section = False

        for line in llm_response.split("\n"):
            lower = line.lower().strip()
            # Detect recommendation section headers
            if any(kw in lower for kw in ["recommend", "action", "next step", "suggestion"]):
                if lower.startswith("#") or lower.startswith("**"):
                    in_rec_section = True
                    continue

            # Collect bullet/numbered items in recommendation sections
            if in_rec_section:
                stripped = line.strip()
                if stripped.startswith(("-", "•", "*")) or re.match(r"^\d+[\.\)]\s", stripped):
                    # Clean the bullet prefix
                    clean = re.sub(r"^[-•*]\s*|^\d+[\.\)]\s*", "", stripped).strip()
                    clean = re.sub(r"\*\*", "", clean)  # remove bold markers
                    if len(clean) > 10:
                        recommendations.append(clean)
                elif stripped == "":
                    in_rec_section = False  # blank line ends section

        return recommendations[:5]  # Cap at 5

    # ------------------------------------------------------------------
    # Existing methods (unchanged from v1)
    # ------------------------------------------------------------------

    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response using Gemini"""
        system_prompt = self.get_system_prompt()

        if context:
            context_str = self._format_context(context)
            full_prompt = f"{system_prompt}\n\n{context_str}\n\n{prompt}"
        else:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            response = self.model.generate_content(full_prompt)
            text = response.text
            text = text.replace("$", "\\$")

            # Downsize headings: convert # and ## to ### and ####
            text = re.sub(r"^# ", "### ", text, flags=re.MULTILINE)
            text = re.sub(r"^## ", "#### ", text, flags=re.MULTILINE)

            self.conversation_history.append({
                "timestamp": datetime.now(),
                "prompt": prompt,
                "response": text,
            })

            return text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Unable to generate response - {str(e)}"

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary into a readable string"""
        lines = ["CONTEXT:"]
        for key, value in context.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info(f"{self.name}: Conversation history cleared")

    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history