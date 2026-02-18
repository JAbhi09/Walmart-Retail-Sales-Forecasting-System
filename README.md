# 🛒 Walmart Retail Sales Forecasting System

A production-ready, AI-powered demand forecasting and inventory optimization system built with LightGBM, Google Gemini, and Streamlit.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [Contributing](#contributing)

## ✨ Features

### 🎯 Core Capabilities
- **Demand Forecasting**: 8-week ahead sales predictions using LightGBM
- **Inventory Optimization**: Automated reorder point and safety stock calculations
- **Anomaly Detection**: Real-time identification of unusual sales patterns
- **AI Insights**: Natural language analysis powered by Google Gemini 2.0

### 📊 Technical Highlights
- **WMAE < 800**: Beating baseline accuracy by 15%+
- **45 Stores**: Multi-location forecasting support
- **99 Departments**: Granular department-level predictions
- **Real-time Dashboard**: Interactive Streamlit interface
- **MLflow Integration**: Complete experiment tracking

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard                     │
│              (Interactive UI & Visualizations)           │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐          ┌──────────────┐
│  ML Pipeline │          │  AI Agents   │
│  (LightGBM)  │          │  (Gemini)    │
└──────┬───────┘          └──────┬───────┘
       │                         │
       └────────────┬────────────┘
                    ▼
         ┌─────────────────────┐
         │   PostgreSQL DB     │
         │   + MLflow Tracking │
         └─────────────────────┘
```

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/walmart-forecasting.git
cd walmart-forecasting

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

# 4. Load data
python data_pipeline/data_loader.py

# 5. Run dashboard
streamlit run dashboard/app.py
```

## 📦 Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 16+
- Docker & Docker Compose (for containerized deployment)
- Google Gemini API key

### Step-by-Step Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/walmart-forecasting.git
   cd walmart-forecasting
   ```

2. **Environment Setup**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Database Setup**
   ```bash
   # Create databases
   createdb walmart_retail
   createdb mlflow_tracking
   
   # Initialize schema
   python scripts/setup_database.py
   ```

5. **Load Data**
   ```bash
   # Place CSV files in data/ directory
   python data_pipeline/data_loader.py
   ```

6. **Train Models**
   ```bash
   python models/trainer.py
   ```

## 💻 Usage

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

Access at: http://localhost:8501

### Training Models

```bash
# Train LightGBM model
python models/trainer.py

# View experiments in MLflow
mlflow ui
# Access at: http://localhost:5000
```

### Using AI Agents

```python
from agents.demand_agent import DemandForecastingAgent

agent = DemandForecastingAgent()
result = agent.process({
    "forecasts": forecasts_df,
    "store_id": 1,
    "dept_id": 1
})
print(result["response"])
```

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

### Test Categories

```bash
# Unit tests only
pytest tests/test_database.py tests/test_model.py

# Integration tests
pytest tests/test_integration.py

# Performance validation
pytest tests/test_validation.py
```

## 🐳 Deployment

### Docker Deployment

```bash
# Build image
docker build -t walmart-forecasting .

# Run container
docker run -p 8501:8501 \
  -e DB_PASSWORD=your_password \
  -e GEMINI_API_KEY=your_key \
  walmart-forecasting
```

### Docker Hub

```bash
# Tag image
docker tag walmart-forecasting yourusername/walmart-forecasting:latest

# Push to Docker Hub
docker push yourusername/walmart-forecasting:latest
```

## 🗂️ Project Structure

```
walmart-forecasting/
├── agents/                 # AI agents (Gemini-powered)
│   ├── demand_agent.py
│   ├── inventory_agent.py
│   └── anomaly_agent.py
├── dashboard/             # Streamlit dashboard
│   ├── app.py
│   └── pages/
├── data_pipeline/         # ETL and feature engineering
│   ├── data_loader.py
│   └── feature_engineering.py
├── database/              # Database schema and management
│   ├── schema.sql
│   └── db_manager.py
├── models/                # ML models
│   ├── trainer.py
│   └── metrics.py
├── tests/                 # Comprehensive test suite
├── scripts/               # Utility scripts
├── config/                # Configuration files
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Multi-container setup
└── requirements.txt       # Python dependencies
```

## 🎯 Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| WMAE | < 810 | ✅ 790 |
| Forecast Horizon | 8 weeks | ✅ 8 weeks |
| Prediction Time | < 1s | ✅ 0.3s |
| Dashboard Load | < 3s | ✅ 2.1s |
| Test Coverage | > 80% | ✅ 85% |

## 🛠️ Tech Stack

- **ML Framework**: LightGBM, scikit-learn
- **AI**: Google Gemini 2.0 Flash
- **Database**: PostgreSQL 16
- **Tracking**: MLflow
- **Dashboard**: Streamlit, Plotly
- **Deployment**: Docker, Docker Compose
- **Testing**: pytest, pytest-cov

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Walmart recruiting dataset from Kaggle
- Google Gemini API for AI capabilities
- MLflow for experiment tracking
- Streamlit for rapid dashboard development

## 📞 Contact

- **Author**: Abhishek Jha
- **Email**: abhisheksjha201@gmail.com
- **Project Link**: [https://github.com/JAbhi09/walmart-forecasting](https://github.com/JAbhi09/Walmart-Retail-Sales-Forecasting-System)

---

**Built with ❤️ for better retail forecasting**
