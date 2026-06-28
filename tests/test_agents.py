"""
Unit tests for AI agents functionality.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from agents.response_model import AgentResponse


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
        assert isinstance(result, AgentResponse)
        assert isinstance(result.llm_response, str)

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
        assert isinstance(result, AgentResponse)
        assert "anomalies_detected" in result.raw_data
        # Our fixture has 2 clear outliers (200k, 250k)
        assert result.raw_data["anomalies_detected"] >= 2

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


# ── validate_dataframe Unit Tests ───────────────────────────

class TestValidateDataframe:
    """Direct tests for the validate_dataframe helper."""

    def test_wrong_type_dict_raises(self):
        from agents.validation import validate_dataframe
        with pytest.raises(ValueError, match="DataFrame"):
            validate_dataframe({"store_id": [1]}, ["store_id"], "test_df")

    def test_none_raises(self):
        from agents.validation import validate_dataframe
        with pytest.raises(ValueError, match="DataFrame"):
            validate_dataframe(None, ["store_id"], "test_df")

    def test_empty_dataframe_raises(self):
        from agents.validation import validate_dataframe
        df = pd.DataFrame({"store_id": pd.Series([], dtype=int)})
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df, ["store_id"], "test_df")

    def test_missing_columns_raises_with_names(self):
        from agents.validation import validate_dataframe
        df = pd.DataFrame({"store_id": [1], "dept_id": [2]})
        with pytest.raises(ValueError, match="missing required column"):
            validate_dataframe(df, ["store_id", "dept_id", "forecast_date"], "test_df")

    def test_missing_columns_error_names_the_absent_ones(self):
        from agents.validation import validate_dataframe
        df = pd.DataFrame({"store_id": [1]})
        with pytest.raises(ValueError, match="forecast_date"):
            validate_dataframe(df, ["store_id", "forecast_date"], "test_df")

    def test_all_null_date_column_raises(self):
        from agents.validation import validate_dataframe
        df = pd.DataFrame({"store_id": [1, 2], "date_col": [None, None]})
        with pytest.raises(ValueError, match="entirely null or unparseable"):
            validate_dataframe(df, ["store_id", "date_col"], "test_df", date_columns=["date_col"])

    def test_unparseable_date_strings_raise(self):
        from agents.validation import validate_dataframe
        df = pd.DataFrame({"store_id": [1, 2], "date_col": ["not-a-date", "also-bad"]})
        with pytest.raises(ValueError, match="entirely null or unparseable"):
            validate_dataframe(df, ["store_id", "date_col"], "test_df", date_columns=["date_col"])

    def test_valid_dataframe_passes(self, sample_forecasts):
        from agents.validation import validate_dataframe
        # Must not raise
        validate_dataframe(
            sample_forecasts,
            ["store_id", "dept_id", "forecast_date", "predicted_sales"],
            "forecasts",
            date_columns=["forecast_date"],
        )

    def test_valid_sales_data_passes(self, sample_sales_data):
        from agents.validation import validate_dataframe
        validate_dataframe(
            sample_sales_data,
            ["store_id", "dept_id", "feature_date", "weekly_sales"],
            "sales_data",
            date_columns=["feature_date"],
        )


# ── Agent Validation Wiring Tests ───────────────────────────

class TestAgentValidationWiring:
    """Verify each agent's entry point rejects bad input with clear errors."""

    # --- DemandForecastingAgent ---

    def test_demand_process_rejects_dict_forecasts(self):
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        with pytest.raises(ValueError, match="DataFrame"):
            agent.process({"forecasts": {"store_id": [1], "dept_id": [2]}})

    def test_demand_process_rejects_missing_columns(self):
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        bad_df = pd.DataFrame({"store_id": [1], "dept_id": [2]})
        with pytest.raises(ValueError, match="missing required column"):
            agent.process({"forecasts": bad_df})

    def test_demand_process_rejects_empty_df(self):
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        empty_df = pd.DataFrame({
            "store_id": pd.Series([], dtype=int),
            "dept_id": pd.Series([], dtype=int),
            "forecast_date": pd.Series([], dtype="datetime64[ns]"),
            "predicted_sales": pd.Series([], dtype=float),
        })
        with pytest.raises(ValueError, match="empty"):
            agent.process({"forecasts": empty_df})

    def test_demand_process_none_forecasts_allowed(self):
        """None forecasts (no data) should not raise — existing behaviour."""
        from agents.demand_agent import DemandForecastingAgent
        agent = DemandForecastingAgent()
        result = agent.process({})
        assert "response" in result

    def test_demand_safe_process_returns_failure_on_bad_input(self):
        from agents.demand_agent import DemandForecastingAgent
        from agents.response_model import AgentStatus
        agent = DemandForecastingAgent()
        bad_df = pd.DataFrame({"store_id": [1]})
        result = agent.safe_process({"forecasts": bad_df})
        assert result.status == AgentStatus.FAILURE
        assert result.error_message is not None
        assert "missing required column" in result.error_message

    # --- InventoryOptimizationAgent ---

    def test_inventory_process_rejects_dict_forecasts(self):
        from agents.inventory_agent import InventoryOptimizationAgent
        agent = InventoryOptimizationAgent()
        with pytest.raises(ValueError, match="DataFrame"):
            agent.process({"forecasts": {"store_id": [1]}})

    def test_inventory_process_rejects_missing_columns(self):
        from agents.inventory_agent import InventoryOptimizationAgent
        agent = InventoryOptimizationAgent()
        bad_df = pd.DataFrame({"store_id": [1], "dept_id": [2]})
        with pytest.raises(ValueError, match="missing required column"):
            agent.process({"forecasts": bad_df})

    def test_inventory_process_rejects_empty_df(self):
        from agents.inventory_agent import InventoryOptimizationAgent
        agent = InventoryOptimizationAgent()
        empty_df = pd.DataFrame({
            "store_id": pd.Series([], dtype=int),
            "dept_id": pd.Series([], dtype=int),
            "forecast_date": pd.Series([], dtype="datetime64[ns]"),
            "predicted_sales": pd.Series([], dtype=float),
        })
        with pytest.raises(ValueError, match="empty"):
            agent.process({"forecasts": empty_df})

    def test_inventory_safe_process_returns_failure_on_bad_input(self):
        from agents.inventory_agent import InventoryOptimizationAgent
        from agents.response_model import AgentStatus
        agent = InventoryOptimizationAgent()
        bad_df = pd.DataFrame({"store_id": [1]})
        result = agent.safe_process({"forecasts": bad_df})
        assert result.status == AgentStatus.FAILURE
        assert "missing required column" in result.error_message

    # --- AnomalyDetectionAgent ---

    def test_anomaly_process_rejects_dict_sales_data(self):
        from agents.anomaly_agent import AnomalyDetectionAgent
        agent = AnomalyDetectionAgent()
        with pytest.raises(ValueError, match="DataFrame"):
            agent.process({"sales_data": {"store_id": [1]}})

    def test_anomaly_process_rejects_missing_columns(self):
        from agents.anomaly_agent import AnomalyDetectionAgent
        agent = AnomalyDetectionAgent()
        bad_df = pd.DataFrame({"store_id": [1], "dept_id": [2]})
        with pytest.raises(ValueError, match="missing required column"):
            agent.process({"sales_data": bad_df})

    def test_anomaly_process_rejects_empty_df(self):
        from agents.anomaly_agent import AnomalyDetectionAgent
        agent = AnomalyDetectionAgent()
        empty_df = pd.DataFrame({
            "store_id": pd.Series([], dtype=int),
            "dept_id": pd.Series([], dtype=int),
            "feature_date": pd.Series([], dtype="datetime64[ns]"),
            "weekly_sales": pd.Series([], dtype=float),
        })
        with pytest.raises(ValueError, match="empty"):
            agent.process({"sales_data": empty_df})

    def test_anomaly_process_none_sales_data_allowed(self):
        """None sales_data should not raise — existing behaviour."""
        from agents.anomaly_agent import AnomalyDetectionAgent
        agent = AnomalyDetectionAgent()
        result = agent.process({})
        assert "response" in result

    def test_anomaly_detect_rejects_dict(self):
        from agents.anomaly_agent import AnomalyDetectionAgent
        agent = AnomalyDetectionAgent()
        with pytest.raises(ValueError, match="DataFrame"):
            agent.detect_anomalies({"store_id": [1]})

    def test_anomaly_detect_rejects_missing_columns(self):
        from agents.anomaly_agent import AnomalyDetectionAgent
        agent = AnomalyDetectionAgent()
        bad_df = pd.DataFrame({"store_id": [1], "dept_id": [2]})
        with pytest.raises(ValueError, match="missing required column"):
            agent.detect_anomalies(bad_df)

    def test_anomaly_detect_rejects_empty_df(self):
        from agents.anomaly_agent import AnomalyDetectionAgent
        agent = AnomalyDetectionAgent()
        empty_df = pd.DataFrame({
            "store_id": pd.Series([], dtype=int),
            "dept_id": pd.Series([], dtype=int),
            "feature_date": pd.Series([], dtype="datetime64[ns]"),
            "weekly_sales": pd.Series([], dtype=float),
        })
        with pytest.raises(ValueError, match="empty"):
            agent.detect_anomalies(empty_df)

    def test_anomaly_safe_process_returns_failure_on_bad_input(self):
        from agents.anomaly_agent import AnomalyDetectionAgent
        from agents.response_model import AgentStatus
        agent = AnomalyDetectionAgent()
        bad_df = pd.DataFrame({"store_id": [1]})
        result = agent.safe_process({"sales_data": bad_df})
        assert result.status == AgentStatus.FAILURE
        assert "missing required column" in result.error_message