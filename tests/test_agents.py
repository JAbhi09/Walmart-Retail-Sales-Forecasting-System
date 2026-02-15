"""
Unit tests for AI agents functionality.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock


# ── Patch Gemini globally so agents can be instantiated without a real API key ──

@pytest.fixture(autouse=True)
def mock_gemini(monkeypatch):
    """Mock Google Generative AI for all tests."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-testing")
    with patch("agents.base_agent.genai") as mock_genai:
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Mocked AI response for testing."
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        yield mock_genai


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def sample_forecasts():
    """Sample forecast DataFrame matching what agents expect."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
    rows = []
    for store in [1, 2]:
        for dept in [1, 2]:
            for d in dates:
                rows.append({
                    "store_id": store,
                    "dept_id": dept,
                    "forecast_date": d,
                    "predicted_sales": np.random.uniform(5000, 50000),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_historical_sales():
    """Sample historical sales DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-06", periods=52, freq="W-FRI")
    rows = []
    for store in [1, 2]:
        for dept in [1, 2]:
            for d in dates:
                rows.append({
                    "store_id": store,
                    "dept_id": dept,
                    "feature_date": d,
                    "weekly_sales": np.random.uniform(5000, 50000),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_sales_data():
    """Sample sales data for anomaly detection."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-06", periods=n, freq="W-FRI")
    return pd.DataFrame({
        "store_id": np.tile([1, 2], n // 2),
        "dept_id": np.tile([1, 2], n // 2),
        "feature_date": dates,
        "weekly_sales": np.concatenate([
            np.random.uniform(5000, 50000, n - 2),
            [200000, 250000],  # outliers
        ]),
    })


# ── BaseAgent Tests ─────────────────────────────────────────

class TestBaseAgent:
    """Test base agent functionality."""

    def test_base_agent_is_abstract(self):
        """BaseAgent cannot be instantiated directly."""
        from agents.base_agent import BaseAgent
        with pytest.raises(TypeError):
            BaseAgent(name="TestAgent")

    def test_concrete_agent_inherits_base(self):
        """Concrete agents inherit from BaseAgent."""
        from agents.base_agent import BaseAgent
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        assert isinstance(agent, BaseAgent)

    def test_agent_has_conversation_history(self):
        """All agents start with empty conversation history."""
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        assert agent.conversation_history == []

    def test_agent_clear_history(self):
        """Agents can clear their conversation history."""
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        agent.conversation_history.append({"prompt": "test", "response": "test"})
        agent.clear_history()
        assert agent.conversation_history == []

    def test_agent_get_history(self):
        """Agents can return their conversation history."""
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        assert agent.get_history() == []

    def test_missing_api_key_raises(self, monkeypatch):
        """Agent raises ValueError when GOOGLE_API_KEY is missing."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        from agents.demand_agent import DemandForecastingAgent
        with patch("agents.base_agent.genai"):
            # Re-import won't help; need to bypass the autouse fixture
            # So we directly test BaseAgent's init guard
            from agents.base_agent import BaseAgent
            monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                # Create a minimal concrete subclass
                class _TestAgent(BaseAgent):
                    def get_system_prompt(self): return ""
                    def process(self, ctx): return {}
                _TestAgent(name="test")


# ── DemandForecastingAgent Tests ────────────────────────────

class TestDemandForecastingAgent:
    """Test demand forecasting agent."""

    def test_initialization(self):
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        assert agent.name == "Demand Forecasting Agent"
        assert agent.model_name == "gemini-2.5-flash"

    def test_get_system_prompt(self):
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        prompt = agent.get_system_prompt()
        assert isinstance(prompt, str)
        assert "Demand Forecasting" in prompt

    def test_process_returns_expected_keys(self, sample_forecasts, sample_historical_sales):
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        result = agent.process({
            "forecasts": sample_forecasts,
            "historical_sales": sample_historical_sales,
            "question": "What is the demand outlook?",
        })
        assert "agent" in result
        assert "response" in result
        assert "timestamp" in result
        assert result["agent"] == "Demand Forecasting Agent"

    def test_process_with_store_filter(self, sample_forecasts):
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        result = agent.process({
            "forecasts": sample_forecasts,
            "store_id": 1,
            "dept_id": 1,
        })
        assert "Store 1" in result["context_summary"]

    def test_summarize_context_all(self):
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        summary = agent._summarize_context({})
        assert summary == "All stores and departments"

    def test_generate_response_appends_history(self):
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        agent.generate_response("Test prompt")
        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0]["prompt"] == "Test prompt"


# ── InventoryOptimizationAgent Tests ────────────────────────

class TestInventoryOptimizationAgent:
    """Test inventory optimization agent."""

    def test_initialization(self):
        from agents.inventory_agent import InventoryOptimizationAgent
        agent = InventoryOptimizationAgent()
        assert agent.name == "Inventory Optimization Agent"

    def test_get_system_prompt(self):
        from agents.inventory_agent import InventoryOptimizationAgent
        agent = InventoryOptimizationAgent()
        prompt = agent.get_system_prompt()
        assert "Inventory Optimization" in prompt

    def test_process_returns_expected_keys(self, sample_forecasts):
        from agents.inventory_agent import InventoryOptimizationAgent
        agent = InventoryOptimizationAgent()
        result = agent.process({
            "forecasts": sample_forecasts,
            "service_level": 0.95,
            "lead_time_days": 7,
        })
        assert "agent" in result
        assert "response" in result
        assert result["agent"] == "Inventory Optimization Agent"

    def test_calculate_safety_stock(self, sample_forecasts):
        from agents.inventory_agent import InventoryOptimizationAgent
        agent = InventoryOptimizationAgent()
        result = agent.calculate_safety_stock(
            forecasts=sample_forecasts,
            service_level=0.95,
            lead_time_days=7,
        )
        assert "response" in result
        assert isinstance(result["response"], str)

    def test_summarize_context_with_service_level(self):
        from agents.inventory_agent import InventoryOptimizationAgent
        agent = InventoryOptimizationAgent()
        summary = agent._summarize_context({
            "store_id": 1,
            "service_level": 0.95,
        })
        assert "Store 1" in summary
        assert "95%" in summary


# ── AnomalyDetectionAgent Tests ────────────────────────────

class TestAnomalyDetectionAgent:
    """Test anomaly detection agent."""

    def test_initialization(self):
        from agents.anomaly_agent import AnomalyDetectionAgent
        agent = AnomalyDetectionAgent()
        assert agent.name == "Anomaly Detection Agent"

    def test_get_system_prompt(self):
        from agents.anomaly_agent import AnomalyDetectionAgent
        agent = AnomalyDetectionAgent()
        prompt = agent.get_system_prompt()
        assert "Anomaly Detection" in prompt

    def test_process_returns_expected_keys(self, sample_sales_data):
        from agents.anomaly_agent import AnomalyDetectionAgent
        agent = AnomalyDetectionAgent()
        result = agent.process({
            "sales_data": sample_sales_data,
            "anomalies": [{"date": "2023-01-06", "store_id": 1, "z_score": "4.5"}],
            "threshold": 3.0,
        })
        assert "agent" in result
        assert "response" in result
        assert "anomalies_detected" in result
        assert result["anomalies_detected"] == 1

    def test_detect_anomalies(self, sample_sales_data):
        from agents.anomaly_agent import AnomalyDetectionAgent
        agent = AnomalyDetectionAgent()
        result = agent.detect_anomalies(sample_sales_data, threshold=3.0)
        assert "response" in result
        assert "anomalies_detected" in result
        # Our fixture has 2 clear outliers (200k, 250k)
        assert result["anomalies_detected"] >= 2

    def test_process_without_anomalies(self, sample_sales_data):
        from agents.anomaly_agent import AnomalyDetectionAgent
        agent = AnomalyDetectionAgent()
        result = agent.process({
            "sales_data": sample_sales_data,
        })
        assert result["anomalies_detected"] == 0


# ── Integration Tests ───────────────────────────────────────

class TestAgentIntegration:
    """Test agent integration and orchestration."""

    def test_all_agents_instantiate(self):
        from agents.demand_agent import DemandForecastingAgent
        from agents.inventory_agent import InventoryOptimizationAgent
        from agents.anomaly_agent import AnomalyDetectionAgent

        agents = [
            DemandForecastingAgent(),
            InventoryOptimizationAgent(),
            AnomalyDetectionAgent(),
        ]
        for agent in agents:
            assert agent is not None
            assert hasattr(agent, "process")
            assert hasattr(agent, "generate_response")

    def test_agents_have_unique_names(self):
        from agents.demand_agent import DemandForecastingAgent
        from agents.inventory_agent import InventoryOptimizationAgent
        from agents.anomaly_agent import AnomalyDetectionAgent

        names = [
            DemandForecastingAgent().name,
            InventoryOptimizationAgent().name,
            AnomalyDetectionAgent().name,
        ]
        assert len(names) == len(set(names)), "Agent names must be unique"

    def test_process_with_empty_context(self):
        """Agents should handle empty context gracefully."""
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        result = agent.process({})
        assert "response" in result