# Walmart Retail Sales Forecasting System

A production-ready, AI-powered demand forecasting and inventory optimization system built with LightGBM, Google Gemini 2.5 Flash, and Streamlit.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Tech Stack](#tech-stack)

## Features

### Core Capabilities

- **Demand Forecasting**: 8-week ahead sales predictions using LightGBM with P10/P90 quantile prediction intervals
- **Inventory Optimization**: Automated reorder point and safety stock calculations (95% service level, 7-day lead time)
- **Anomaly Detection**: Real-time identification of unusual sales patterns using a 3-sigma threshold
- **AI Insights**: Natural language analysis powered by Google Gemini 2.5 Flash
- **Cross-Agent Synthesis**: Contradictions and connections identified across all three agents that no single agent would catch alone
- **Critical Alerts Roll-up**: Aggregated critical-severity findings surfaced for dashboards or notifications

### Technical Highlights

- **WMAE < 800**: Beating baseline accuracy by 15%+
- **45 Stores**: Multi-location forecasting support
- **99 Departments**: Granular department-level predictions
- **Real-time Dashboard**: Interactive Streamlit interface
- **MLflow Integration**: Complete experiment tracking with model registry
- **Structured Agent Responses**: Typed `AgentResponse` model with status, insights, recommendations, and execution timing
- **Per-Agent Error Isolation**: One agent failure does not stop the pipeline — remaining agents still return results

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard                     │
│              (dashboard/Home.py)                         │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐          ┌────────────────────┐
│  ML Pipeline │          │  AgentOrchestrator  │
│  (LightGBM)  │          │                    │
│  + Quantile  │          │  ┌──────────────┐  │
│  Regression  │          │  │ DemandAgent  │  │
└──────┬───────┘          │  ├──────────────┤  │
       │                  │  │ InventoryAgt │  │
       │                  │  ├──────────────┤  │
       │                  │  │ AnomalyAgent │  │
       │                  │  └──────────────┘  │
       │                  │  Cross-Agent Synth  │
       │                  └──────────┬─────────┘
       └────────────┬────────────────┘
                    ▼
         ┌─────────────────────┐
         │   PostgreSQL DB     │
         │   + MLflow Tracking │
         └─────────────────────┘
```

### Multi-Agent System

The `AgentOrchestrator` runs three specialized agents with per-agent error isolation via `safe_process()`, which returns a structured `AgentResponse`:

| Agent | Role |
|---|---|
| `DemandForecastingAgent` | Trend analysis, demand change %, seasonal patterns |
| `InventoryOptimizationAgent` | Safety stock, reorder points, demand coefficient of variation |
| `AnomalyDetectionAgent` | 3-sigma outlier detection on historical sales |

After all three complete, the orchestrator runs **cross-agent synthesis** to surface:
- Contradictions (e.g., demand decline vs. safety stock calculated from higher-demand periods)
- Dependency warnings (e.g., anomalies skewing forecast training data)
- Reinforcing signals (e.g., high demand CV confirmed by both inventory and anomaly agents)

## Quick Start

### Using Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/JAbhi09/Walmart-Retail-Sales-Forecasting-System.git
cd Walmart-Retail-Sales-Forecasting-System

# 2. Create .env file
cp .env.example .env
# Edit .env with your credentials

# 3. Start all services
docker-compose up -d

# 4. Access the dashboard
open http://localhost:8501
```

### Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up database
python scripts/setup_database.py

# 4. Load and engineer features
python data_pipeline/data_loader.py
python data_pipeline/run_feature_engineering.py

# 5. Train model
python models/trainer.py

# 6. Run dashboard
streamlit run dashboard/Home.py
```

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 16+
- Docker & Docker Compose (for containerized deployment)
- Google Gemini API key (`GOOGLE_API_KEY`)

### Step-by-Step

1. **Clone Repository**
   ```bash
   git clone https://github.com/JAbhi09/Walmart-Retail-Sales-Forecasting-System.git
   ```

2. **Environment Setup**
   ```bash
   cp .env.example .env
   # Set GOOGLE_API_KEY, DB_PASSWORD, MLFLOW_TRACKING_URI
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Database Setup**
   ```bash
   createdb walmart_retail
   createdb mlflow_db
   python scripts/setup_database.py
   ```

5. **Load Data**
   ```bash
   # Place train.csv, test.csv, stores.csv, features.csv in data/
   python data_pipeline/data_loader.py
   python data_pipeline/run_feature_engineering.py
   ```

6. **Train Model**
   ```bash
   python models/trainer.py
   ```

## Usage

### Running the Dashboard

```bash
streamlit run dashboard/Home.py
```

Access at: `http://localhost:8501`

### Training Models

```bash
# Train LightGBM model (point forecast + P10/P90 quantile models)
python models/trainer.py

# View experiments in MLflow
mlflow ui
# Access at: http://localhost:5000
```

### Using AI Agents Programmatically

```python
from agents.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()

result = orchestrator.analyze_forecast(
    forecasts=forecasts_df,
    historical_sales=historical_df,
    store_id=1,
    dept_id=1,
)

print(result["summary"])
print(result["critical_alerts"])
print(result["cross_agent_synthesis"])

# Ask a single agent directly
from agents.demand_agent import DemandForecastingAgent

agent = DemandForecastingAgent()
response = agent.safe_process({
    "forecasts": forecasts_df,
    "historical_sales": historical_df,
    "store_id": 1,
    "question": "What are the key demand trends for next 8 weeks?",
})
print(response.summary)
print(response.recommendations)
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=. --cov-report=html
# Open htmlcov/index.html
```

### Test Categories

```bash
# Unit tests
pytest tests/test_model.py tests/test_database.py tests/test_feature_engineering.py

# Agent tests
pytest tests/test_agents.py tests/test_validation.py

# Integration tests
pytest tests/test_integration.py
```

## Deployment

### Docker

```bash
docker build -t walmart-forecasting .

docker run -p 8501:8501 \
  -e GOOGLE_API_KEY=your_key \
  -e DB_PASSWORD=your_password \
  -e MLFLOW_TRACKING_URI=your_uri \
  walmart-forecasting
```

### Docker Compose (full stack)

```bash
docker-compose up -d
```

Spins up: Streamlit dashboard, PostgreSQL, MLflow tracking server.

## Project Structure

```
walmart-forecasting/
├── agents/
│   ├── base_agent.py          # Abstract base with safe_process() and Gemini integration
│   ├── demand_agent.py        # Demand trend analysis
│   ├── inventory_agent.py     # Safety stock and reorder optimization
│   ├── anomaly_agent.py       # 3-sigma outlier detection
│   ├── orchestrator.py        # Multi-agent coordinator + cross-agent synthesis
│   ├── response_model.py      # AgentResponse, AgentStatus, Insight, InsightSeverity
│   ├── validation.py          # Input DataFrame validation
│   └── __init__.py
├── dashboard/
│   └── Home.py                # Streamlit entry point
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   ├── features.csv           # Engineered features
├── data_pipeline/
│   ├── data_loader.py         # ETL from CSV to PostgreSQL
│   ├── feature_engineering.py # Lag, rolling, temporal, and store features
│   └── run_feature_engineering.py
├── database/
│   ├── schema.sql             # PostgreSQL schema
│   ├── db_manager.py          # Connection management
│   └── create-mlflow-db.sh
├── models/
│   ├── trainer.py             # WalmartForecaster: train, predict, save, load
│   ├── train.py               # Training entry script
│   ├── metrics.py             # WMAE, MAE, RMSE
│   └── generate_forecasts.py  # Batch forecast generation
├── scripts/
│   ├── setup_database.py
│   ├── train_model.py
│   ├── test_agents.py
│   ├── run_tests.py
│   └── deploy.py
├── tests/
│   ├── test_agents.py
│   ├── test_database.py
│   ├── test_feature_engineering.py
│   ├── test_integration.py
│   ├── test_model.py
│   ├── test_validation.py
│   └── conftest.py
├── config/
│   └── config.yaml            # Model params, feature config, thresholds
├── utils/
│   └── logger.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Performance Metrics

| Metric | Target | Achieved |
|---|---|---|
| WMAE | < 810 | 790 |
| Forecast Horizon | 8 weeks | 8 weeks |
| Prediction Time | < 1s | 0.3s |
| Dashboard Load | < 3s | 2.1s |
| Test Coverage | > 80% | 85% |

### Model Configuration

| Parameter | Value |
|---|---|
| Boosting rounds | 2000 (early stopping at 100) |
| Learning rate | 0.03 |
| Num leaves | 127 |
| Quantile bounds | P10 / P90 |
| Holiday weight (WMAE) | 5× vs regular weeks |

### Feature Engineering

- **Lag features**: 1, 2, 4, 8, 52 weeks
- **Rolling windows**: 4-week (mean, std, min, max), 13-week (mean, std), 52-week (mean, std)
- **Temporal**: week of year, month, quarter, month start/end flags
- **Store**: store type (one-hot), normalized size

## Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | LightGBM, scikit-learn |
| AI / LLM | Google Gemini 2.5 Flash |
| Database | PostgreSQL 16 |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit, Plotly |
| Deployment | Docker, Docker Compose |
| Testing | pytest, pytest-cov |
| Language | Python 3.11 |

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

- **Author**: Abhishek Jha
- **Email**: abhisheksjha201@gmail.com
- **Project**: [github.com/JAbhi09/Walmart-Retail-Sales-Forecasting-System](https://github.com/JAbhi09/Walmart-Retail-Sales-Forecasting-System)
