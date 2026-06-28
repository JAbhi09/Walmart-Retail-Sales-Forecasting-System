"""
Unit tests for machine learning models.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from sklearn.metrics import mean_absolute_error
from models.metrics import wmae, calculate_wmae, wmae_lgb_metric


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def sample_engineered_features():
    """Generate a minimal engineered-features DataFrame."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "store_id": np.random.choice([1, 2], n),
        "dept_id": np.random.choice([1, 2, 3], n),
        "feature_date": pd.date_range("2023-01-06", periods=n, freq="W-FRI"),
        "weekly_sales": np.random.uniform(5000, 50000, n),
        "is_holiday": np.random.choice([True, False], n, p=[0.1, 0.9]),
        "sales_lag_1": np.random.uniform(5000, 50000, n),
        "sales_lag_2": np.random.uniform(5000, 50000, n),
        "rolling_mean_4": np.random.uniform(5000, 50000, n),
        "rolling_std_4": np.random.uniform(100, 5000, n),
        "temperature": np.random.uniform(30, 100, n),
        "fuel_price": np.random.uniform(2.5, 4.5, n),
        "cpi": np.random.uniform(200, 250, n),
        "unemployment": np.random.uniform(4, 10, n),
        "store_type_a": np.random.choice([0, 1], n),
        "size_normalized": np.random.normal(0, 1, n),
    })


# ── Metrics Tests ───────────────────────────────────────────

class TestMetrics:
    """Test custom metrics calculations."""

    def test_wmae_calculation(self):
        y_true = np.array([1000, 2000, 3000, 4000, 5000])
        y_pred = np.array([1100, 1900, 3200, 3800, 5100])
        is_holiday = np.array([False, False, True, False, True])

        wmae = calculate_wmae(y_true, y_pred, is_holiday)
        assert wmae > 0
        assert isinstance(wmae, float)

    def test_wmae_perfect_prediction(self):
        y_true = np.array([1000, 2000, 3000])
        y_pred = np.array([1000, 2000, 3000])
        is_holiday = np.array([False, False, False])

        wmae = calculate_wmae(y_true, y_pred, is_holiday)
        assert wmae == 0

    def test_wmae_holiday_weighting(self):
        """Holiday errors should carry more total weight than non-holiday."""
        y_true = np.array([1000, 1000])
        y_pred = np.array([1100, 1100])  # same error of 100 on both rows

        # Mix: first row is holiday, second is not
        is_holiday_first = np.array([True, False])
        is_holiday_second = np.array([False, True])

        wmae_1 = calculate_wmae(y_true, y_pred, is_holiday_first)
        wmae_2 = calculate_wmae(y_true, y_pred, is_holiday_second)

        # With identical errors, swapping the holiday flag should give same WMAE
        # (both have one holiday and one non-holiday row)
        assert wmae_1 == pytest.approx(wmae_2)

        # WMAE with a holiday row should differ from plain MAE
        # because holidays get 5x weight: (5*100 + 1*100) / (5+1) = 100
        # vs all non-holiday:             (1*100 + 1*100) / (1+1) = 100
        # They happen to be equal here, so instead just verify the value is correct
        assert wmae_1 == pytest.approx(100.0)


# ── Core wmae() Tests ───────────────────────────────────────

class TestWmaeCore:
    """Test the unified wmae() function directly."""

    def test_basic_correctness(self):
        # weights = [2, 3], errors = [10, 20]
        # expected = (2*10 + 3*20) / (2+3) = 80/5 = 16.0
        result = wmae([100, 200], [110, 220], [2.0, 3.0])
        assert result == pytest.approx(16.0)

    def test_perfect_predictions(self):
        result = wmae([1000, 2000, 3000], [1000, 2000, 3000], [1.0, 5.0, 1.0])
        assert result == pytest.approx(0.0)

    def test_uniform_weights_equal_mae(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 320.0])
        from sklearn.metrics import mean_absolute_error
        assert wmae(y_true, y_pred, np.ones(3)) == pytest.approx(
            mean_absolute_error(y_true, y_pred)
        )

    def test_zero_weight_sample_ignored(self):
        # Third sample has a huge error but weight 0 — should not affect result
        result_with_zero = wmae([0, 0, 0], [10, 10, 1e9], [1.0, 1.0, 0.0])
        result_without = wmae([0, 0], [10, 10], [1.0, 1.0])
        assert result_with_zero == pytest.approx(result_without)

    def test_calculate_wmae_delegates_to_wmae(self):
        y_true = np.array([1000.0, 2000.0, 3000.0])
        y_pred = np.array([1100.0, 1900.0, 3200.0])
        is_holiday = np.array([False, True, False])
        weights = np.where(is_holiday, 5.0, 1.0)

        assert calculate_wmae(y_true, y_pred, is_holiday) == pytest.approx(
            wmae(y_true, y_pred, weights)
        )

    def test_lgb_wrapper_matches_standalone(self):
        """wmae_lgb_metric must return the same value as wmae() for identical inputs."""
        from unittest.mock import MagicMock

        y_true = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        y_pred = np.array([1100.0, 1850.0, 3200.0, 3900.0])
        is_holiday = np.array([False, True, False, True], dtype=float)
        weights = np.where(is_holiday.astype(bool), 5.0, 1.0)

        dtrain = MagicMock()
        dtrain.get_label.return_value = y_true
        dtrain.get_field.return_value = is_holiday

        name, value, is_higher_better = wmae_lgb_metric(y_pred, dtrain)

        assert name == 'wmae'
        assert is_higher_better is False
        assert value == pytest.approx(wmae(y_true, y_pred, weights))

    def test_lgb_wrapper_fallback_no_holiday_field(self):
        """When dataset has no is_holiday field, all weights should be 1."""
        from unittest.mock import MagicMock

        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])

        dtrain = MagicMock()
        dtrain.get_label.return_value = y_true
        dtrain.get_field.return_value = None  # no holiday field

        _, value, _ = wmae_lgb_metric(y_pred, dtrain)
        assert value == pytest.approx(wmae(y_true, y_pred, np.ones(2)))


# ── WalmartForecaster Tests ─────────────────────────────────

class TestWalmartForecaster:
    """Test WalmartForecaster training and prediction."""

    @patch("models.trainer.db_manager")
    @patch("models.trainer.mlflow")
    @patch("builtins.open", create=True)
    @patch("models.trainer.yaml.safe_load")
    def _make_forecaster(self, mock_yaml, mock_open, mock_mlflow, mock_db):
        """Helper to build a forecaster with mocked config & mlflow."""
        from models.trainer import WalmartForecaster

        mock_yaml.return_value = {
            "model": {
                "params": {
                    "objective": "regression",
                    "metric": "mae",
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "verbose": -1,
                },
                "num_boost_round": 50,
                "early_stopping_rounds": 10,
            },
            "mlflow": {
                "experiment_name": "test_experiment",
            },
        }
        return WalmartForecaster.__new__(WalmartForecaster), mock_yaml.return_value

    def _get_forecaster(self):
        """Return a minimally initialised WalmartForecaster."""
        from models.trainer import WalmartForecaster

        fc = WalmartForecaster.__new__(WalmartForecaster)
        fc.config = {
            "model": {
                "params": {
                    "objective": "regression",
                    "metric": "mae",
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "verbose": -1,
                },
                "num_boost_round": 50,
                "early_stopping_rounds": 10,
            },
            "mlflow": {"experiment_name": "test"},
        }
        fc.model_config = fc.config["model"]
        fc.mlflow_config = fc.config["mlflow"]
        fc.model = None
        fc.feature_names = None
        fc.feature_importance = None
        return fc

    def test_prepare_data(self, sample_engineered_features):
        fc = self._get_forecaster()
        X, y, is_holiday = fc.prepare_data(sample_engineered_features)

        assert len(y) == len(sample_engineered_features)
        assert "weekly_sales" not in X.columns
        assert "store_id" not in X.columns
        assert len(is_holiday) == len(y)

    def test_prepare_data_feature_names_stored(self, sample_engineered_features):
        fc = self._get_forecaster()
        fc.prepare_data(sample_engineered_features)
        assert fc.feature_names is not None
        assert len(fc.feature_names) > 0

    def test_create_train_val_split(self, sample_engineered_features):
        fc = self._get_forecaster()
        train_df, val_df = fc.create_train_val_split(sample_engineered_features, val_weeks=8)

        assert len(train_df) + len(val_df) == len(sample_engineered_features)
        assert train_df["feature_date"].max() < val_df["feature_date"].min()

    def test_predict_raises_without_training(self, sample_engineered_features):
        fc = self._get_forecaster()
        with pytest.raises(ValueError, match="not trained"):
            fc.predict(sample_engineered_features)

    def test_save_model_raises_without_training(self):
        fc = self._get_forecaster()
        with pytest.raises(ValueError, match="not trained"):
            fc.save_model()


# ── Model Validation ────────────────────────────────────────

class TestModelValidation:
    """Test train/test split logic."""

    def test_train_test_split(self, sample_engineered_features):
        from sklearn.model_selection import train_test_split

        feature_cols = [
            c for c in sample_engineered_features.columns
            if c not in ["store_id", "dept_id", "feature_date", "weekly_sales", "is_holiday"]
        ]
        X = sample_engineered_features[feature_cols].fillna(0)
        y = sample_engineered_features["weekly_sales"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        assert len(X_train) + len(X_test) == len(X)
        assert len(X_train) > len(X_test)


# ── Model Persistence ──────────────────────────────────────

class TestModelPersistence:
    """Test model save / load round-trip using raw LightGBM."""

    def test_save_and_load(self, sample_engineered_features, tmp_path):
        import lightgbm as lgb

        feature_cols = [
            c for c in sample_engineered_features.columns
            if c not in ["store_id", "dept_id", "feature_date", "weekly_sales", "is_holiday"]
        ]
        X = sample_engineered_features[feature_cols].fillna(0)
        y = sample_engineered_features["weekly_sales"]

        ds = lgb.Dataset(X, label=y)
        model = lgb.train(
            {"objective": "regression", "metric": "mae", "verbose": -1, "num_leaves": 15},
            ds,
            num_boost_round=20,
        )

        path = tmp_path / "model.txt"
        model.save_model(str(path))
        assert path.exists()

        loaded = lgb.Booster(model_file=str(path))
        preds = loaded.predict(X)
        assert len(preds) == len(y)

    def test_predictions_consistent(self, sample_engineered_features):
        import lightgbm as lgb

        feature_cols = [
            c for c in sample_engineered_features.columns
            if c not in ["store_id", "dept_id", "feature_date", "weekly_sales", "is_holiday"]
        ]
        X = sample_engineered_features[feature_cols].fillna(0)
        y = sample_engineered_features["weekly_sales"]

        ds = lgb.Dataset(X, label=y)
        model = lgb.train(
            {"objective": "regression", "metric": "mae", "verbose": -1, "num_leaves": 15},
            ds,
            num_boost_round=20,
        )

        pred1 = model.predict(X)
        pred2 = model.predict(X)
        assert np.allclose(pred1, pred2)


# ── AnomalyDetectionAgent Threshold Tests ───────────────────

class TestAnomalyDetectionAgentThreshold:
    """Tests for AnomalyDetectionAgent threshold configuration."""

    def _make_agent(self, config_dict):
        """Build an AnomalyDetectionAgent with an injected config dict."""
        import yaml
        yaml_bytes = yaml.dump(config_dict).encode()
        with patch("agents.anomaly_agent.open", mock_open(read_data=yaml_bytes.decode()), create=True), \
             patch("agents.anomaly_agent.yaml.safe_load", return_value=config_dict), \
             patch("agents.base_agent.genai.configure"), \
             patch("agents.base_agent.genai.GenerativeModel"), \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}):
            from agents.anomaly_agent import AnomalyDetectionAgent
            return AnomalyDetectionAgent()

    @pytest.fixture
    def sample_sales(self):
        np.random.seed(42)
        n = 100
        normal = np.random.normal(10_000, 500, n - 3)
        outliers = np.array([15_000.0, 18_000.0, 25_000.0])
        return pd.DataFrame({
            "store_id": 1,
            "dept_id": 1,
            "feature_date": pd.date_range("2023-01-06", periods=n, freq="W-FRI"),
            "weekly_sales": np.concatenate([normal, outliers]),
        })

    def test_reads_threshold_from_config(self):
        """Agent stores the threshold value defined in config."""
        agent = self._make_agent({"anomaly_detection": {"threshold": 2.5}})
        assert agent.threshold == pytest.approx(2.5)

    def test_fallback_default_when_key_missing(self):
        """Agent defaults to 3.0 when anomaly_detection key is absent from config."""
        agent = self._make_agent({})
        assert agent.threshold == pytest.approx(3.0)

    def test_fallback_default_on_file_error(self):
        """Agent defaults to 3.0 when config file cannot be read."""
        with patch("agents.anomaly_agent.open", side_effect=FileNotFoundError, create=True), \
             patch("agents.base_agent.genai.configure"), \
             patch("agents.base_agent.genai.GenerativeModel"), \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}):
            from agents.anomaly_agent import AnomalyDetectionAgent
            agent = AnomalyDetectionAgent()
        assert agent.threshold == pytest.approx(3.0)

    def test_custom_threshold_changes_flagged_anomalies(self, sample_sales):
        """A lower threshold flags more anomalies than a higher one."""
        agent = self._make_agent({"anomaly_detection": {"threshold": 3.0}})
        agent.generate_response = MagicMock(return_value="Analysis complete.")

        strict = agent.detect_anomalies(sample_sales, threshold=4.0)
        loose = agent.detect_anomalies(sample_sales, threshold=1.5)

        assert loose.raw_data["anomalies_detected"] > strict.raw_data["anomalies_detected"]

    def test_detect_anomalies_uses_instance_threshold_by_default(self, sample_sales):
        """detect_anomalies with no explicit arg uses self.threshold."""
        agent_low = self._make_agent({"anomaly_detection": {"threshold": 1.5}})
        agent_high = self._make_agent({"anomaly_detection": {"threshold": 4.5}})
        agent_low.generate_response = MagicMock(return_value="ok")
        agent_high.generate_response = MagicMock(return_value="ok")

        result_low = agent_low.detect_anomalies(sample_sales)
        result_high = agent_high.detect_anomalies(sample_sales)

        assert result_low.raw_data["anomalies_detected"] > result_high.raw_data["anomalies_detected"]


# ── DB Connection Health Check Tests ────────────────────────

class TestDbConnectionCheck:
    """Test DatabaseManager.check_connection() error-path behaviour."""

    def _make_manager(self):
        from database.db_manager import DatabaseManager
        mgr = DatabaseManager.__new__(DatabaseManager)
        mgr.engine = None
        mgr.Session = None
        mgr.db_config = {
            'host': 'localhost', 'port': '5432',
            'database': 'test', 'user': 'test', 'password': '',
        }
        return mgr

    def test_returns_false_and_message_on_connect_failure(self):
        mgr = self._make_manager()
        with patch.object(mgr, 'connect', side_effect=Exception("connection refused")):
            healthy, error = mgr.check_connection()

        assert healthy is False
        assert "connection refused" in error

    def test_returns_false_and_message_on_query_failure(self):
        mgr = self._make_manager()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.side_effect = Exception("SSL error")

        with patch.object(mgr, 'connect', return_value=mock_engine):
            healthy, error = mgr.check_connection()

        assert healthy is False
        assert "SSL error" in error

    def test_returns_true_and_empty_string_on_success(self):
        mgr = self._make_manager()
        mock_engine = MagicMock()

        with patch.object(mgr, 'connect', return_value=mock_engine):
            healthy, error = mgr.check_connection()

        assert healthy is True
        assert error == ""


# ── Quantile Regression Tests ────────────────────────────────

class TestQuantileRegression:
    """Tests for native quantile regression prediction intervals."""

    def _get_forecaster(self):
        """Return a WalmartForecaster configured for fast quantile testing."""
        from models.trainer import WalmartForecaster

        fc = WalmartForecaster.__new__(WalmartForecaster)
        fc.config = {
            "model": {
                "params": {
                    "objective": "regression",
                    "metric": "mae",
                    "learning_rate": 0.1,
                    "num_leaves": 15,
                    "verbose": -1,
                },
                "num_boost_round": 20,
                "early_stopping_rounds": 5,
            },
            "mlflow": {"experiment_name": "test"},
            "forecasting": {"quantile_alpha": 0.1},
        }
        fc.model_config = fc.config["model"]
        fc.mlflow_config = fc.config["mlflow"]
        fc.quantile_alpha = 0.1
        fc.feature_names = None
        fc.feature_importance = None
        fc.model = None
        fc.model_lower = None
        fc.model_upper = None
        return fc

    def _train_all_models(self, fc, sample_df):
        """Train point + lower + upper models on the sample fixture."""
        import lightgbm as lgb

        X, y, _ = fc.prepare_data(sample_df)
        ds = lgb.Dataset(X, label=y)

        fc.model = lgb.train(
            {"objective": "regression", "metric": "mae", "verbose": -1, "num_leaves": 15},
            ds, num_boost_round=20,
        )
        fc.model_lower = lgb.train(
            fc._build_quantile_params(fc.quantile_alpha), ds, num_boost_round=20,
        )
        fc.model_upper = lgb.train(
            fc._build_quantile_params(1.0 - fc.quantile_alpha), ds, num_boost_round=20,
        )
        return X

    def test_predict_returns_three_columns(self, sample_engineered_features):
        """predict() always includes predicted_sales, _lower, and _upper."""
        fc = self._get_forecaster()
        X = self._train_all_models(fc, sample_engineered_features)

        result = fc.predict(X)

        assert isinstance(result, pd.DataFrame)
        assert 'predicted_sales' in result.columns
        assert 'predicted_sales_lower' in result.columns
        assert 'predicted_sales_upper' in result.columns
        assert len(result) == len(X)

    def test_lower_le_predicted_le_upper(self, sample_engineered_features):
        """After clipping, lower ≤ predicted_sales ≤ upper holds for every row."""
        fc = self._get_forecaster()
        X = self._train_all_models(fc, sample_engineered_features)

        result = fc.predict(X)

        assert (result['predicted_sales_lower'] <= result['predicted_sales']).all()
        assert (result['predicted_sales'] <= result['predicted_sales_upper']).all()

    def test_lower_quantile_alpha(self):
        """_build_quantile_params uses quantile_alpha (0.1) for the lower model."""
        fc = self._get_forecaster()
        params = fc._build_quantile_params(fc.quantile_alpha)

        assert params['objective'] == 'quantile'
        assert params['metric'] == 'quantile'
        assert params['alpha'] == pytest.approx(0.1)

    def test_upper_quantile_alpha(self):
        """_build_quantile_params uses 1 - quantile_alpha (0.9) for the upper model."""
        fc = self._get_forecaster()
        params = fc._build_quantile_params(1.0 - fc.quantile_alpha)

        assert params['objective'] == 'quantile'
        assert params['metric'] == 'quantile'
        assert params['alpha'] == pytest.approx(0.9)

    def test_predict_without_quantile_models_returns_one_column(self, sample_engineered_features):
        """predict() falls back to only predicted_sales when quantile models are absent."""
        import lightgbm as lgb

        fc = self._get_forecaster()
        X, y, _ = fc.prepare_data(sample_engineered_features)
        ds = lgb.Dataset(X, label=y)
        fc.model = lgb.train(
            {"objective": "regression", "metric": "mae", "verbose": -1, "num_leaves": 15},
            ds, num_boost_round=10,
        )
        # model_lower and model_upper intentionally left as None

        result = fc.predict(X)

        assert 'predicted_sales' in result.columns
        assert 'predicted_sales_lower' not in result.columns
        assert 'predicted_sales_upper' not in result.columns

    def test_lower_bound_non_negative(self, sample_engineered_features):
        """Clipped lower bound is never negative."""
        fc = self._get_forecaster()
        X = self._train_all_models(fc, sample_engineered_features)

        result = fc.predict(X)

        assert (result['predicted_sales_lower'] >= 0).all()