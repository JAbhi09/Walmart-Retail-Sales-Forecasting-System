# Multi-Agent Retail Demand Forecasting & Inventory Optimization System
## Complete Project Documentation

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Prerequisites & Environment Setup](#3-prerequisites--environment-setup)
4. [Database Design & Setup](#4-database-design--setup)
5. [Data Engineering Pipeline](#5-data-engineering-pipeline)
6. [Machine Learning Models](#6-machine-learning-models)
7. [Multi-Agent System](#7-multi-agent-system)
8. [Dashboard Development](#8-dashboard-development)
9. [Integration & Testing](#9-integration--testing)
10. [Deployment](#10-deployment)
11. [Troubleshooting](#11-troubleshooting)
12. [Resources & References](#12-resources--references)

---

## 1. Project Overview

### 1.1 Problem Statement

Walmart faces significant challenges in demand forecasting and inventory management:
- **$1.77 trillion** lost globally due to inventory distortion (stockouts + overstocking)
- **43%** of consumers switch to competitors when items are out of stock
- Seasonal demand variations cause revenue unpredictability
- Poor inventory planning leads to capital locked in excess stock

### 1.2 Solution

A Multi-Agent AI system that:
- Predicts sales 8 weeks in advance using machine learning
- Optimizes inventory with automated reorder point calculations
- Detects anomalies in sales patterns
- Generates actionable business insights through AI agents

### 1.3 Key Features

- **Demand Forecasting**: LightGBM + Prophet models for accurate predictions
- **Inventory Optimization**: Automated safety stock and reorder point calculations
- **Anomaly Detection**: Real-time identification of unusual sales patterns
- **AI Agents**: CrewAI-powered agents for intelligent analysis
- **Interactive Dashboard**: Streamlit-based visualization and control center

### 1.4 Expected Outcomes

- Forecast accuracy: WMAE < 700 (beating baseline of 821)
- Reduce stockouts by 15%
- Reduce overstock by 20%
- Automate 80% of routine inventory decisions

### 1.5 Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Python | 3.11 | Core programming |
| Data Processing | PySpark | 3.5+ | ETL pipeline |
| Database | PostgreSQL | 16 | Data storage |
| ML Framework | LightGBM | 4.x | Primary forecasting model |
| Baseline Model | Prophet | 1.1.5 | Time series baseline |
| ML Tracking | MLflow | 2.17+ | Experiment management |
| AI Agents | CrewAI | 0.152+ | Multi-agent orchestration |
| LLM | Gemini 2.0 Flash | Latest | Agent intelligence |
| Dashboard | Streamlit | 1.38+ | User interface |
| Visualization | Plotly | Latest | Interactive charts |
| Containers | Docker | Latest | Deployment |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                     │
│                    (Streamlit Dashboard)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ├─── Reads ───┐
                         │             │
                         ▼             ▼
┌──────────────────────────────┐  ┌─────────────────────────┐
│    PRESENTATION LAYER         │  │   AI AGENT LAYER        │
│  - Forecasts Page             │  │  - CrewAI Orchestrator  │
│  - Inventory Page             │  │  - 4 Specialized Agents │
│  - AI Insights Page           │  │  - Gemini 2.0 Flash LLM │
└────────────┬─────────────────┘  └───────────┬─────────────┘
             │                                 │
             │ Reads/Writes                    │ Reads/Writes
             ▼                                 ▼
┌──────────────────────────────────────────────────────────────┐
│                    DATA STORAGE LAYER                         │
│                     (PostgreSQL Database)                     │
│  - raw_sales          - forecasts                            │
│  - external_features  - inventory_recommendations            │
│  - engineered_features                                       │
└────────────┬─────────────────────────────────────────────────┘
             │
             │ Feeds
             ▼
┌──────────────────────────────────────────────────────────────┐
│                  MACHINE LEARNING LAYER                       │
│  - LightGBM Model     - Prophet Model     - MLflow Registry  │
└────────────┬─────────────────────────────────────────────────┘
             │
             │ Trained on
             ▼
┌──────────────────────────────────────────────────────────────┐
│                DATA PROCESSING LAYER                          │
│                   (PySpark ETL Pipeline)                      │
│  - Extract    - Transform    - Load                          │
└────────────┬─────────────────────────────────────────────────┘
             │
             │ Ingests
             ▼
┌──────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                               │
│  train.csv    features.csv    stores.csv                     │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Diagram

```
Raw CSV Files
    │
    ├─► PySpark ETL Pipeline
    │       │
    │       ├─► Data Cleaning
    │       ├─► Feature Engineering
    │       │   ├─► Lag Features (7, 14, 28, 52 days)
    │       │   ├─► Rolling Statistics (4, 13, 52 weeks)
    │       │   ├─► Time Features (week, month, holiday)
    │       │   └─► Economic Features (CPI, unemployment)
    │       │
    │       └─► Output
    │           ├─► PostgreSQL (engineered_features table)
    │           └─► Parquet Files (for ML training)
    │
    ├─► Machine Learning Training
    │       │
    │       ├─► LightGBM Model
    │       │   ├─► Train on 80% data
    │       │   ├─► Validate on 20% data
    │       │   └─► Log to MLflow
    │       │
    │       ├─► Prophet Model
    │       │   ├─► Train per store
    │       │   └─► Log to MLflow
    │       │
    │       └─► Generate Forecasts
    │           └─► Write to PostgreSQL (forecasts table)
    │
    ├─► AI Agent Processing
    │       │
    │       ├─► Anomaly Detection Agent
    │       │   └─► Validates data quality
    │       │
    │       ├─► Forecasting Agent
    │       │   ├─► Loads model from MLflow
    │       │   ├─► Generates predictions
    │       │   └─► Explains trends via Gemini
    │       │
    │       ├─► Inventory Optimization Agent
    │       │   ├─► Calculates reorder points
    │       │   ├─► Determines safety stock
    │       │   └─► Identifies stockout risk
    │       │
    │       └─► Report Writer Agent
    │           └─► Synthesizes insights
    │
    └─► Streamlit Dashboard
            │
            ├─► Displays Forecasts
            ├─► Shows Inventory Status
            └─► Presents AI Insights
```

### 2.3 Component Interaction

```
User Action: "Show forecast for Store 1, Dept 38"
    │
    ▼
Streamlit Dashboard
    │
    ├─► Query: SELECT * FROM forecasts 
    │          WHERE store_id=1 AND dept_id=38
    │
    └─► PostgreSQL returns: Forecast data
            │
            └─► Display: Plotly chart with predictions

User Action: "Get AI Analysis"
    │
    ▼
Streamlit triggers CrewAI Flow
    │
    ├─► Agent 1: Anomaly Detection
    │   ├─► Tool: query_recent_sales()
    │   ├─► PostgreSQL: Returns last 52 weeks
    │   └─► Gemini API: Analyzes patterns
    │
    ├─► Agent 2: Forecasting
    │   ├─► Tool: load_model_from_mlflow()
    │   ├─► MLflow: Returns model
    │   ├─► Tool: prepare_features()
    │   ├─► PostgreSQL: Returns feature data
    │   └─► Gemini API: Explains predictions
    │
    ├─► Agent 3: Inventory Optimization
    │   ├─► Reads forecast from Agent 2
    │   ├─► Tool: calculate_reorder_point()
    │   └─► Gemini API: Generates recommendations
    │
    └─► Agent 4: Report Writer
        ├─► Reads all agent outputs
        ├─► Gemini API: Synthesizes report
        └─► PostgreSQL: Saves report
            │
            └─► Streamlit displays: Formatted insights
```

---

## 3. Prerequisites & Environment Setup

### 3.1 System Requirements

**Minimum Requirements:**
- OS: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- RAM: 8 GB (16 GB recommended)
- Storage: 10 GB free space
- CPU: 4 cores
- Internet: Required for API calls and package installation

**Software Prerequisites:**
- Python 3.11.x
- Java 11 or 17 (for PySpark)
- PostgreSQL 16
- Docker Desktop
- Git

### 3.2 Installation Guide

#### 3.2.1 Python Installation

**Check existing installation:**
```bash
python --version
# or
python3 --version
```

**If needed, install Python 3.11:**
- Windows: Download from [python.org](https://www.python.org/downloads/)
- macOS: `brew install python@3.11`
- Linux: `sudo apt-get install python3.11`

#### 3.2.2 Java Installation (for PySpark)

**Check existing installation:**
```bash
java -version
```

**Install if needed:**
- Download Eclipse Temurin from [adoptium.net](https://adoptium.net/)
- Choose Java 11 LTS or Java 17 LTS
- Add to PATH after installation

**Verify installation:**
```bash
java -version
# Should show: openjdk version "11.x.x" or "17.x.x"
```

#### 3.2.3 PostgreSQL Installation

**You already have this ✅**

**Verify:**
- Open pgAdmin4
- Connect to local server
- Should see default `postgres` database

#### 3.2.4 Docker Desktop Installation

**Download:**
- Windows/Mac: [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
- Linux: Install Docker Engine

**Verify installation:**
```bash
docker --version
docker-compose --version
```

**Start Docker Desktop and keep it running**

#### 3.2.5 Git Installation

**Check existing installation:**
```bash
git --version
```

**Install if needed:**
- Download from [git-scm.com](https://git-scm.com/downloads)

### 3.3 Project Setup

#### 3.3.1 Create Project Directory

```bash
# Create main project folder
mkdir walmart-forecasting
cd walmart-forecasting

# Create subfolder structure
mkdir -p data/raw data/processed data/models
mkdir -p etl models agents streamlit_app database notebooks
mkdir -p agents/tools streamlit_app/pages streamlit_app/components
```

#### 3.3.2 Initialize Git Repository

```bash
git init
```

**Create .gitignore file:**
```bash
# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Data
data/raw/*.csv
data/processed/*.parquet
*.pkl
*.joblib

# Secrets
.env
*.key

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# MLflow
mlruns/
mlflow.db

# OS
.DS_Store
Thumbs.db

# Docker
docker-compose.override.yml
EOF
```

#### 3.3.3 Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Verify activation (should show venv in prompt)
which python  # Should show path to venv/bin/python
```

#### 3.3.4 Create Requirements File

**Create `requirements.txt`:**
```txt
# Data Processing
pyspark==3.5.1
pandas==2.1.4
numpy==1.26.3
pyarrow==14.0.2

# Machine Learning
lightgbm==4.2.0
prophet==1.1.5
scikit-learn==1.4.0
xgboost==2.0.3

# MLflow
mlflow==2.17.0
psycopg2-binary==2.9.9

# CrewAI and LLM
crewai[tools]==0.152.0
google-generativeai==0.3.2
langchain==0.1.16
chromadb==0.4.22

# Dashboard
streamlit==1.38.0
plotly==5.18.0
sqlalchemy==2.0.25

# Utilities
python-dotenv==1.0.0
pydantic==2.5.3
jupyter==1.0.0
```

**Install all dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**This will take 5-10 minutes. Expected output:**
```
Successfully installed crewai-0.152.0 streamlit-1.38.0 ...
```

#### 3.3.5 Verify Installations

**Create a test script `verify_setup.py`:**
```python
import sys

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError:
        print(f"❌ {module_name}")
        return False

print("Checking Python version...")
print(f"Python {sys.version}")
print()

print("Checking required packages...")
packages = [
    'pyspark', 'pandas', 'numpy', 'lightgbm', 'prophet',
    'mlflow', 'crewai', 'google.generativeai', 'streamlit',
    'plotly', 'sqlalchemy', 'psycopg2', 'dotenv'
]

results = [check_import(pkg) for pkg in packages]

print()
if all(results):
    print("🎉 All packages installed successfully!")
else:
    print("⚠️  Some packages missing. Check errors above.")
```

**Run verification:**
```bash
python verify_setup.py
```

### 3.4 API Keys and Credentials

#### 3.4.1 Get Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API Key"
3. Sign in with Google account
4. Click "Create API Key"
5. Copy the key (starts with `AIza...`)

#### 3.4.2 Create .env File

**Create `.env` in project root:**
```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=walmart_forecasting
DB_USER=postgres
DB_PASSWORD=your_postgres_password

# MLflow Database
MLFLOW_DB_HOST=localhost
MLFLOW_DB_PORT=5432
MLFLOW_DB_NAME=mlflow_tracking
MLFLOW_DB_USER=postgres
MLFLOW_DB_PASSWORD=your_postgres_password

# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# MLflow Tracking
MLFLOW_TRACKING_URI=http://localhost:5000

# PySpark Configuration
SPARK_HOME=/path/to/spark
PYSPARK_PYTHON=python
```

**⚠️ Important:** Replace `your_postgres_password` and `your_gemini_api_key_here` with actual values.

### 3.5 Dataset Placement

**Copy your downloaded CSV files:**
```bash
# Copy files to data/raw/
cp /path/to/downloaded/train.csv data/raw/
cp /path/to/downloaded/features.csv data/raw/
cp /path/to/downloaded/stores.csv data/raw/
```

**Verify files are there:**
```bash
ls -lh data/raw/
# Should show:
# train.csv (40 MB)
# features.csv (400 KB)
# stores.csv (2 KB)
```

---

## 4. Database Design & Setup

### 4.1 Database Creation

#### 4.1.1 Create Databases in pgAdmin4

**Steps:**
1. Open pgAdmin4
2. Connect to PostgreSQL server
3. Right-click "Databases" → Create → Database
   - Name: `walmart_forecasting`
   - Owner: `postgres`
4. Repeat for second database:
   - Name: `mlflow_tracking`
   - Owner: `postgres`

#### 4.1.2 Verify Database Creation

**In pgAdmin4:**
- Expand "Databases"
- You should see:
  - `postgres` (default)
  - `walmart_forecasting` ✅
  - `mlflow_tracking` ✅

### 4.2 Database Schema

#### 4.2.1 Schema Overview

```
walmart_forecasting
│
├── raw_sales                  (421,570 rows)
│   ├── store_id (PK)
│   ├── dept_id (PK)
│   ├── sale_date (PK)
│   ├── weekly_sales
│   └── is_holiday
│
├── store_attributes           (45 rows)
│   ├── store_id (PK)
│   ├── store_type
│   └── size_sqft
│
├── external_features          (6,435 rows)
│   ├── store_id (PK)
│   ├── feature_date (PK)
│   ├── temperature
│   ├── fuel_price
│   ├── markdown_1 to markdown_5
│   ├── cpi
│   ├── unemployment
│   └── is_holiday
│
├── engineered_features        (421,570 rows)
│   ├── store_id (PK)
│   ├── dept_id (PK)
│   ├── feature_date (PK)
│   ├── [all lag features]
│   ├── [all rolling features]
│   ├── [all time features]
│   └── [all economic features]
│
├── forecasts                  (variable)
│   ├── forecast_id (PK)
│   ├── store_id
│   ├── dept_id
│   ├── forecast_date
│   ├── predicted_sales
│   ├── prediction_lower
│   ├── prediction_upper
│   ├── model_name
│   └── created_at
│
├── inventory_recommendations  (variable)
│   ├── recommendation_id (PK)
│   ├── store_id
│   ├── dept_id
│   ├── recommendation_date
│   ├── current_stock
│   ├── reorder_point
│   ├── safety_stock
│   ├── days_until_stockout
│   ├── status
│   └── created_at
│
└── agent_outputs              (variable)
    ├── output_id (PK)
    ├── store_id
    ├── dept_id
    ├── analysis_type
    ├── agent_response
    └── created_at
```

#### 4.2.2 Create Schema SQL Script

**Create `database/schema.sql`:**
```sql
-- ============================================
-- Walmart Forecasting Database Schema
-- ============================================

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS agent_outputs CASCADE;
DROP TABLE IF EXISTS inventory_recommendations CASCADE;
DROP TABLE IF EXISTS forecasts CASCADE;
DROP TABLE IF EXISTS engineered_features CASCADE;
DROP TABLE IF EXISTS external_features CASCADE;
DROP TABLE IF EXISTS raw_sales CASCADE;
DROP TABLE IF EXISTS store_attributes CASCADE;

-- ============================================
-- 1. Store Attributes (Static Reference Data)
-- ============================================
CREATE TABLE store_attributes (
    store_id INTEGER PRIMARY KEY,
    store_type VARCHAR(1) NOT NULL CHECK (store_type IN ('A', 'B', 'C')),
    size_sqft INTEGER NOT NULL CHECK (size_sqft > 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_store_type ON store_attributes(store_type);

COMMENT ON TABLE store_attributes IS 'Static store characteristics';
COMMENT ON COLUMN store_attributes.store_type IS 'Store category: A (largest), B (medium), C (smallest)';
COMMENT ON COLUMN store_attributes.size_sqft IS 'Store size in square feet';

-- ============================================
-- 2. Raw Sales Data (Training Data)
-- ============================================
CREATE TABLE raw_sales (
    store_id INTEGER NOT NULL,
    dept_id INTEGER NOT NULL,
    sale_date DATE NOT NULL,
    weekly_sales DECIMAL(10, 2) NOT NULL CHECK (weekly_sales >= 0),
    is_holiday BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (store_id, dept_id, sale_date),
    FOREIGN KEY (store_id) REFERENCES store_attributes(store_id)
);

CREATE INDEX idx_sales_date ON raw_sales(sale_date);
CREATE INDEX idx_sales_store_dept ON raw_sales(store_id, dept_id);
CREATE INDEX idx_sales_holiday ON raw_sales(is_holiday);

COMMENT ON TABLE raw_sales IS 'Historical weekly sales by store and department';

-- ============================================
-- 3. External Features (Environmental & Economic)
-- ============================================
CREATE TABLE external_features (
    store_id INTEGER NOT NULL,
    feature_date DATE NOT NULL,
    temperature DECIMAL(5, 2),
    fuel_price DECIMAL(4, 3),
    markdown_1 DECIMAL(10, 2) DEFAULT 0,
    markdown_2 DECIMAL(10, 2) DEFAULT 0,
    markdown_3 DECIMAL(10, 2) DEFAULT 0,
    markdown_4 DECIMAL(10, 2) DEFAULT 0,
    markdown_5 DECIMAL(10, 2) DEFAULT 0,
    cpi DECIMAL(12, 7),
    unemployment DECIMAL(5, 3),
    is_holiday BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (store_id, feature_date),
    FOREIGN KEY (store_id) REFERENCES store_attributes(store_id)
);

CREATE INDEX idx_features_date ON external_features(feature_date);

COMMENT ON TABLE external_features IS 'Time-varying external factors by store';
COMMENT ON COLUMN external_features.cpi IS 'Consumer Price Index';
COMMENT ON COLUMN external_features.markdown_1 IS 'Promotional markdown amount (NULL = no markdown)';

-- ============================================
-- 4. Engineered Features (ML-Ready Data)
-- ============================================
CREATE TABLE engineered_features (
    store_id INTEGER NOT NULL,
    dept_id INTEGER NOT NULL,
    feature_date DATE NOT NULL,
    
    -- Time features
    week_of_year INTEGER CHECK (week_of_year BETWEEN 1 AND 53),
    month INTEGER CHECK (month BETWEEN 1 AND 12),
    quarter INTEGER CHECK (quarter BETWEEN 1 AND 4),
    is_month_start BOOLEAN,
    is_month_end BOOLEAN,
    is_quarter_start BOOLEAN,
    is_quarter_end BOOLEAN,
    days_to_holiday INTEGER,
    
    -- Lag features
    sales_lag_1 DECIMAL(10, 2),
    sales_lag_2 DECIMAL(10, 2),
    sales_lag_4 DECIMAL(10, 2),
    sales_lag_8 DECIMAL(10, 2),
    sales_lag_52 DECIMAL(10, 2),
    
    -- Rolling window features
    rolling_mean_4 DECIMAL(10, 2),
    rolling_mean_13 DECIMAL(10, 2),
    rolling_mean_52 DECIMAL(10, 2),
    rolling_std_4 DECIMAL(10, 2),
    rolling_std_13 DECIMAL(10, 2),
    rolling_min_4 DECIMAL(10, 2),
    rolling_max_4 DECIMAL(10, 2),
    
    -- Economic features
    temperature DECIMAL(5, 2),
    temperature_deviation DECIMAL(5, 2),
    fuel_price DECIMAL(4, 3),
    fuel_price_change DECIMAL(4, 3),
    cpi DECIMAL(12, 7),
    cpi_change DECIMAL(12, 7),
    unemployment DECIMAL(5, 3),
    unemployment_change DECIMAL(5, 3),
    
    -- Markdown features
    total_markdown DECIMAL(10, 2) DEFAULT 0,
    has_markdown BOOLEAN DEFAULT FALSE,
    markdown_count INTEGER DEFAULT 0,
    
    -- Store features
    store_type VARCHAR(1),
    size_sqft INTEGER,
    size_normalized DECIMAL(5, 4),
    
    -- Target variable
    weekly_sales DECIMAL(10, 2),
    is_holiday BOOLEAN,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (store_id, dept_id, feature_date),
    FOREIGN KEY (store_id) REFERENCES store_attributes(store_id)
);

CREATE INDEX idx_eng_features_date ON engineered_features(feature_date);
CREATE INDEX idx_eng_features_store_dept ON engineered_features(store_id, dept_id);

COMMENT ON TABLE engineered_features IS 'Fully processed features for ML model training';

-- ============================================
-- 5. Forecasts (Model Predictions)
-- ============================================
CREATE TABLE forecasts (
    forecast_id SERIAL PRIMARY KEY,
    store_id INTEGER NOT NULL,
    dept_id INTEGER NOT NULL,
    forecast_date DATE NOT NULL,
    predicted_sales DECIMAL(10, 2) NOT NULL,
    prediction_lower DECIMAL(10, 2),
    prediction_upper DECIMAL(10, 2),
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),
    confidence_score DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (store_id) REFERENCES store_attributes(store_id),
    UNIQUE (store_id, dept_id, forecast_date, model_name)
);

CREATE INDEX idx_forecast_date ON forecasts(forecast_date);
CREATE INDEX idx_forecast_store_dept ON forecasts(store_id, dept_id);
CREATE INDEX idx_forecast_model ON forecasts(model_name);

COMMENT ON TABLE forecasts IS 'ML model predictions for future sales';

-- ============================================
-- 6. Inventory Recommendations (Agent Outputs)
-- ============================================
CREATE TABLE inventory_recommendations (
    recommendation_id SERIAL PRIMARY KEY,
    store_id INTEGER NOT NULL,
    dept_id INTEGER NOT NULL,
    recommendation_date DATE NOT NULL,
    
    -- Current state
    current_stock INTEGER,
    avg_daily_demand DECIMAL(10, 2),
    demand_volatility DECIMAL(10, 2),
    
    -- Recommendations
    reorder_point DECIMAL(10, 2) NOT NULL,
    safety_stock DECIMAL(10, 2) NOT NULL,
    order_quantity DECIMAL(10, 2),
    
    -- Risk metrics
    days_until_stockout DECIMAL(5, 2),
    stockout_probability DECIMAL(5, 4),
    status VARCHAR(20) CHECK (status IN ('adequate', 'monitor', 'reorder', 'critical')),
    
    -- AI-generated insights
    agent_reasoning TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (store_id) REFERENCES store_attributes(store_id)
);

CREATE INDEX idx_inventory_date ON inventory_recommendations(recommendation_date);
CREATE INDEX idx_inventory_status ON inventory_recommendations(status);
CREATE INDEX idx_inventory_store_dept ON inventory_recommendations(store_id, dept_id);

COMMENT ON TABLE inventory_recommendations IS 'AI agent inventory optimization recommendations';

-- ============================================
-- 7. Agent Outputs (Audit Trail)
-- ============================================
CREATE TABLE agent_outputs (
    output_id SERIAL PRIMARY KEY,
    store_id INTEGER,
    dept_id INTEGER,
    analysis_type VARCHAR(50) NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    agent_response TEXT NOT NULL,
    execution_time_seconds DECIMAL(8, 3),
    tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (store_id) REFERENCES store_attributes(store_id)
);

CREATE INDEX idx_agent_outputs_type ON agent_outputs(analysis_type);
CREATE INDEX idx_agent_outputs_agent ON agent_outputs(agent_name);
CREATE INDEX idx_agent_outputs_created ON agent_outputs(created_at DESC);

COMMENT ON TABLE agent_outputs IS 'Log of all AI agent analyses for audit and debugging';

-- ============================================
-- Views for Common Queries
-- ============================================

-- View: Latest forecasts by store and department
CREATE VIEW latest_forecasts AS
SELECT 
    f.*,
    sa.store_type,
    sa.size_sqft
FROM forecasts f
JOIN store_attributes sa ON f.store_id = sa.store_id
WHERE f.created_at = (
    SELECT MAX(created_at) 
    FROM forecasts f2 
    WHERE f2.store_id = f.store_id 
    AND f2.dept_id = f.dept_id 
    AND f2.model_name = f.model_name
);

-- View: Critical inventory alerts
CREATE VIEW critical_inventory AS
SELECT 
    ir.*,
    sa.store_type,
    sa.size_sqft
FROM inventory_recommendations ir
JOIN store_attributes sa ON ir.store_id = sa.store_id
WHERE ir.status IN ('reorder', 'critical')
AND ir.recommendation_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY ir.days_until_stockout ASC;

-- View: Sales performance summary
CREATE VIEW sales_summary AS
SELECT 
    rs.store_id,
    sa.store_type,
    COUNT(DISTINCT rs.dept_id) as dept_count,
    SUM(rs.weekly_sales) as total_sales,
    AVG(rs.weekly_sales) as avg_weekly_sales,
    MIN(rs.sale_date) as first_sale_date,
    MAX(rs.sale_date) as last_sale_date
FROM raw_sales rs
JOIN store_attributes sa ON rs.store_id = sa.store_id
GROUP BY rs.store_id, sa.store_type;

-- ============================================
-- Grant Permissions
-- ============================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO postgres;

-- ============================================
-- Success Message
-- ============================================
DO $$
BEGIN
    RAISE NOTICE '✅ Database schema created successfully!';
    RAISE NOTICE 'Tables created: 7';
    RAISE NOTICE 'Views created: 3';
    RAISE NOTICE 'Ready for data ingestion.';
END $$;
```

#### 4.2.3 Execute Schema Creation

**In pgAdmin4:**
1. Select `walmart_forecasting` database
2. Click Tools → Query Tool
3. Copy the entire `schema.sql` content
4. Paste into Query Tool
5. Click Execute (F5)
6. Verify: Should see "✅ Database schema created successfully!"

**Verify tables created:**
```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;
```

**Expected output:**
- agent_outputs
- engineered_features
- external_features
- forecasts
- inventory_recommendations
- raw_sales
- store_attributes

### 4.3 Database Connection Testing

**Create `database/test_connection.py`:**
```python
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def test_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        
        print("✅ Database connection successful!")
        print(f"PostgreSQL version: {db_version[0]}")
        
        # Test table creation
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        
        print(f"\n📊 Tables found: {len(tables)}")
        for table in tables:
            print(f"  - {table[0]}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_connection()
```

**Run test:**
```bash
python database/test_connection.py
```

---

## 5. Data Engineering Pipeline

### 5.1 ETL Pipeline Overview

**Goal:** Transform raw CSV files into ML-ready features in PostgreSQL

**Pipeline Stages:**
1. **Extract**: Read CSV files with PySpark
2. **Transform**: Clean, join, and engineer features
3. **Load**: Write to PostgreSQL and Parquet

### 5.2 Extract Module

**Create `etl/extract.py`:**
```python
"""
Extract module: Read CSV files into PySpark DataFrames
"""
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_spark_session(app_name="WalmartETL"):
    """Initialize Spark session with optimized configuration"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    
    logger.info(f"✅ Spark session created: {app_name}")
    return spark

def read_training_data(spark, path="data/raw/train.csv"):
    """
    Read training data (Store, Dept, Date, Weekly_Sales, IsHoliday)
    
    Expected schema:
    - Store: int
    - Dept: int
    - Date: string (MM-DD-YYYY format)
    - Weekly_Sales: float
    - IsHoliday: boolean
    """
    logger.info(f"Reading training data from {path}")
    
    schema = StructType([
        StructField("Store", IntegerType(), False),
        StructField("Dept", IntegerType(), False),
        StructField("Date", StringType(), False),
        StructField("Weekly_Sales", FloatType(), False),
        StructField("IsHoliday", BooleanType(), False)
    ])
    
    df = spark.read.csv(path, header=True, schema=schema)
    
    # Convert date string to proper date type
    from pyspark.sql.functions import to_date
    df = df.withColumn("Date", to_date("Date", "dd-MM-yyyy"))
    
    logger.info(f"✅ Training data loaded: {df.count()} rows")
    return df

def read_features_data(spark, path="data/raw/features.csv"):
    """
    Read features data (Store, Date, Temperature, Fuel_Price, etc.)
    """
    logger.info(f"Reading features data from {path}")
    
    schema = StructType([
        StructField("Store", IntegerType(), False),
        StructField("Date", StringType(), False),
        StructField("Temperature", FloatType(), True),
        StructField("Fuel_Price", FloatType(), True),
        StructField("MarkDown1", FloatType(), True),
        StructField("MarkDown2", FloatType(), True),
        StructField("MarkDown3", FloatType(), True),
        StructField("MarkDown4", FloatType(), True),
        StructField("MarkDown5", FloatType(), True),
        StructField("CPI", FloatType(), True),
        StructField("Unemployment", FloatType(), True),
        StructField("IsHoliday", BooleanType(), False)
    ])
    
    df = spark.read.csv(path, header=True, schema=schema)
    
    # Convert date
    from pyspark.sql.functions import to_date
    df = df.withColumn("Date", to_date("Date", "dd-MM-yyyy"))
    
    # Fill NA markdowns with 0
    from pyspark.sql.functions import coalesce, lit
    for i in range(1, 6):
        col_name = f"MarkDown{i}"
        df = df.withColumn(col_name, coalesce(df[col_name], lit(0.0)))
    
    logger.info(f"✅ Features data loaded: {df.count()} rows")
    return df

def read_stores_data(spark, path="data/raw/stores.csv"):
    """
    Read stores data (Store, Type, Size)
    """
    logger.info(f"Reading stores data from {path}")
    
    schema = StructType([
        StructField("Store", IntegerType(), False),
        StructField("Type", StringType(), False),
        StructField("Size", IntegerType(), False)
    ])
    
    df = spark.read.csv(path, header=True, schema=schema)
    
    logger.info(f"✅ Stores data loaded: {df.count()} rows")
    return df

def validate_data(df, name):
    """Basic data quality checks"""
    logger.info(f"Validating {name}")
    
    # Check for nulls
    from pyspark.sql.functions import col, count, when
    null_counts = df.select([
        count(when(col(c).isNull(), c)).alias(c) 
        for c in df.columns
    ])
    
    logger.info(f"Null counts for {name}:")
    null_counts.show()
    
    # Check row count
    total_rows = df.count()
    logger.info(f"Total rows: {total_rows}")
    
    return True
```

### 5.3 Transform Module

**Create `etl/transform.py`:**
```python
"""
Transform module: Feature engineering and data preparation
"""
from pyspark.sql import Window
from pyspark.sql.functions import *
import logging

logger = logging.getLogger(__name__)

def join_all_tables(train_df, features_df, stores_df):
    """
    Join training, features, and stores data
    """
    logger.info("Joining all tables")
    
    # Join training with features on Store and Date
    df = train_df.join(
        features_df,
        on=["Store", "Date"],
        how="left"
    )
    
    # Join with stores on Store
    df = df.join(
        stores_df,
        on="Store",
        how="left"
    )
    
    logger.info(f"✅ Tables joined: {df.count()} rows")
    return df

def create_time_features(df):
    """
    Extract time-based features from date
    """
    logger.info("Creating time features")
    
    df = df.withColumn("week_of_year", weekofyear("Date"))
    df = df.withColumn("month", month("Date"))
    df = df.withColumn("quarter", quarter("Date"))
    df = df.withColumn("year", year("Date"))
    df = df.withColumn("day_of_week", dayofweek("Date"))
    df = df.withColumn("day_of_month", dayofmonth("Date"))
    df = df.withColumn("day_of_year", dayofyear("Date"))
    
    # Boolean flags
    df = df.withColumn("is_month_start", 
                       when(col("day_of_month") <= 7, True).otherwise(False))
    df = df.withColumn("is_month_end", 
                       when(col("day_of_month") >= 24, True).otherwise(False))
    df = df.withColumn("is_quarter_start", 
                       when(col("month").isin([1,4,7,10]), True).otherwise(False))
    df = df.withColumn("is_quarter_end", 
                       when(col("month").isin([3,6,9,12]), True).otherwise(False))
    
    logger.info("✅ Time features created")
    return df

def create_lag_features(df):
    """
    Create lag features for sales
    CRITICAL: Must be ordered by date within each store-dept group
    """
    logger.info("Creating lag features")
    
    # Define window: partition by store and dept, order by date
    window_spec = Window.partitionBy("Store", "Dept").orderBy("Date")
    
    # Create lags for 1, 2, 4, 8, and 52 weeks
    df = df.withColumn("sales_lag_1", lag("Weekly_Sales", 1).over(window_spec))
    df = df.withColumn("sales_lag_2", lag("Weekly_Sales", 2).over(window_spec))
    df = df.withColumn("sales_lag_4", lag("Weekly_Sales", 4).over(window_spec))
    df = df.withColumn("sales_lag_8", lag("Weekly_Sales", 8).over(window_spec))
    df = df.withColumn("sales_lag_52", lag("Weekly_Sales", 52).over(window_spec))
    
    logger.info("✅ Lag features created")
    return df

def create_rolling_features(df):
    """
    Create rolling window statistics
    """
    logger.info("Creating rolling features")
    
    # Windows for different periods
    window_4 = Window.partitionBy("Store", "Dept") \
                     .orderBy("Date") \
                     .rowsBetween(-3, 0)
    
    window_13 = Window.partitionBy("Store", "Dept") \
                      .orderBy("Date") \
                      .rowsBetween(-12, 0)
    
    window_52 = Window.partitionBy("Store", "Dept") \
                      .orderBy("Date") \
                      .rowsBetween(-51, 0)
    
    # 4-week rolling statistics
    df = df.withColumn("rolling_mean_4", avg("Weekly_Sales").over(window_4))
    df = df.withColumn("rolling_std_4", stddev("Weekly_Sales").over(window_4))
    df = df.withColumn("rolling_min_4", min("Weekly_Sales").over(window_4))
    df = df.withColumn("rolling_max_4", max("Weekly_Sales").over(window_4))
    
    # 13-week (quarterly) rolling statistics
    df = df.withColumn("rolling_mean_13", avg("Weekly_Sales").over(window_13))
    df = df.withColumn("rolling_std_13", stddev("Weekly_Sales").over(window_13))
    
    # 52-week (yearly) rolling mean
    df = df.withColumn("rolling_mean_52", avg("Weekly_Sales").over(window_52))
    
    logger.info("✅ Rolling features created")
    return df

def create_economic_features(df):
    """
    Create derived features from economic indicators
    """
    logger.info("Creating economic features")
    
    # Window for computing changes
    window_spec = Window.partitionBy("Store").orderBy("Date")
    
    # Temperature deviation from historical average
    df = df.withColumn("temp_rolling_avg", 
                       avg("Temperature").over(window_spec.rowsBetween(-52, 0)))
    df = df.withColumn("temperature_deviation", 
                       col("Temperature") - col("temp_rolling_avg"))
    
    # Fuel price change
    df = df.withColumn("fuel_price_prev", 
                       lag("Fuel_Price", 1).over(window_spec))
    df = df.withColumn("fuel_price_change", 
                       col("Fuel_Price") - col("fuel_price_prev"))
    
    # CPI change
    df = df.withColumn("cpi_prev", lag("CPI", 4).over(window_spec))
    df = df.withColumn("cpi_change", col("CPI") - col("cpi_prev"))
    
    # Unemployment change
    df = df.withColumn("unemployment_prev", 
                       lag("Unemployment", 4).over(window_spec))
    df = df.withColumn("unemployment_change", 
                       col("Unemployment") - col("unemployment_prev"))
    
    # Clean up intermediate columns
    df = df.drop("temp_rolling_avg", "fuel_price_prev", 
                 "cpi_prev", "unemployment_prev")
    
    logger.info("✅ Economic features created")
    return df

def create_markdown_features(df):
    """
    Create aggregated markdown features
    """
    logger.info("Creating markdown features")
    
    # Total markdown amount
    df = df.withColumn("total_markdown",
                       col("MarkDown1") + col("MarkDown2") + 
                       col("MarkDown3") + col("MarkDown4") + 
                       col("MarkDown5"))
    
    # Has any markdown (boolean)
    df = df.withColumn("has_markdown",
                       when(col("total_markdown") > 0, True).otherwise(False))
    
    # Count of active markdowns
    df = df.withColumn("markdown_count",
                       (when(col("MarkDown1") > 0, 1).otherwise(0) +
                        when(col("MarkDown2") > 0, 1).otherwise(0) +
                        when(col("MarkDown3") > 0, 1).otherwise(0) +
                        when(col("MarkDown4") > 0, 1).otherwise(0) +
                        when(col("MarkDown5") > 0, 1).otherwise(0)))
    
    logger.info("✅ Markdown features created")
    return df

def create_store_features(df):
    """
    Create features from store attributes
    """
    logger.info("Creating store features")
    
    # One-hot encode store type
    df = df.withColumn("store_type_A", 
                       when(col("Type") == "A", 1).otherwise(0))
    df = df.withColumn("store_type_B", 
                       when(col("Type") == "B", 1).otherwise(0))
    df = df.withColumn("store_type_C", 
                       when(col("Type") == "C", 1).otherwise(0))
    
    # Normalize store size
    max_size = df.agg(max("Size")).collect()[0][0]
    df = df.withColumn("size_normalized", col("Size") / max_size)
    
    logger.info("✅ Store features created")
    return df

def clean_feature_names(df):
    """
    Rename columns to snake_case for database compatibility
    """
    logger.info("Cleaning column names")
    
    column_mapping = {
        "Store": "store_id",
        "Dept": "dept_id",
        "Date": "feature_date",
        "Weekly_Sales": "weekly_sales",
        "IsHoliday": "is_holiday",
        "Temperature": "temperature",
        "Fuel_Price": "fuel_price",
        "MarkDown1": "markdown_1",
        "MarkDown2": "markdown_2",
        "MarkDown3": "markdown_3",
        "MarkDown4": "markdown_4",
        "MarkDown5": "markdown_5",
        "CPI": "cpi",
        "Unemployment": "unemployment",
        "Type": "store_type",
        "Size": "size_sqft"
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)
    
    logger.info("✅ Column names cleaned")
    return df

def remove_null_targets(df):
    """
    Remove rows where target (weekly_sales) is null
    This happens for lag features at the start of time series
    """
    logger.info("Removing rows with null targets")
    
    initial_count = df.count()
    df = df.filter(col("weekly_sales").isNotNull())
    final_count = df.count()
    
    logger.info(f"Removed {initial_count - final_count} rows with null targets")
    logger.info(f"✅ Final dataset: {final_count} rows")
    
    return df

def run_full_transformation(train_df, features_df, stores_df):
    """
    Run complete transformation pipeline
    """
    logger.info("=== Starting Full Transformation Pipeline ===")
    
    # 1. Join all tables
    df = join_all_tables(train_df, features_df, stores_df)
    
    # 2. Create time features
    df = create_time_features(df)
    
    # 3. Create lag features
    df = create_lag_features(df)
    
    # 4. Create rolling features
    df = create_rolling_features(df)
    
    # 5. Create economic features
    df = create_economic_features(df)
    
    # 6. Create markdown features
    df = create_markdown_features(df)
    
    # 7. Create store features
    df = create_store_features(df)
    
    # 8. Clean column names
    df = clean_feature_names(df)
    
    # 9. Remove null targets
    df = remove_null_targets(df)
    
    logger.info("=== Transformation Pipeline Complete ===")
    
    return df
```

### 5.4 Load Module

**Create `etl/load.py`:**
```python
"""
Load module: Write processed data to PostgreSQL and Parquet
"""
import logging
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)

def get_jdbc_properties():
    """
    Get JDBC connection properties from environment variables
    """
    return {
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "driver": "org.postgresql.Driver"
    }

def get_jdbc_url():
    """
    Build JDBC URL from environment variables
    """
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    database = os.getenv("DB_NAME")
    return f"jdbc:postgresql://{host}:{port}/{database}"

def write_to_postgres(df, table_name, mode="overwrite"):
    """
    Write DataFrame to PostgreSQL table
    
    Args:
        df: PySpark DataFrame
        table_name: Name of target table
        mode: Write mode ('overwrite', 'append', 'ignore')
    """
    logger.info(f"Writing to PostgreSQL table: {table_name}")
    
    jdbc_url = get_jdbc_url()
    properties = get_jdbc_properties()
    
    df.write.jdbc(
        url=jdbc_url,
        table=table_name,
        mode=mode,
        properties=properties
    )
    
    row_count = df.count()
    logger.info(f"✅ Wrote {row_count} rows to {table_name}")

def write_to_parquet(df, path, mode="overwrite"):
    """
    Write DataFrame to Parquet file
    
    Args:
        df: PySpark DataFrame
        path: Output path for Parquet file
        mode: Write mode ('overwrite', 'append')
    """
    logger.info(f"Writing to Parquet: {path}")
    
    df.write.parquet(path, mode=mode)
    
    logger.info(f"✅ Wrote Parquet file to {path}")

def load_processed_data(df):
    """
    Load processed features to both PostgreSQL and Parquet
    """
    logger.info("=== Starting Data Load ===")
    
    # Write to PostgreSQL
    write_to_postgres(df, "engineered_features", mode="overwrite")
    
    # Write to Parquet (for ML training)
    write_to_parquet(df, "data/processed/features.parquet", mode="overwrite")
    
    logger.info("=== Data Load Complete ===")
```

### 5.5 Main ETL Pipeline

**Create `etl/run_pipeline.py`:**
```python
"""
Main ETL pipeline orchestrator
Executes: Extract → Transform → Load
"""
import logging
from extract import create_spark_session, read_training_data, read_features_data, read_stores_data, validate_data
from transform import run_full_transformation
from load import load_processed_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Run complete ETL pipeline
    """
    try:
        logger.info("="*60)
        logger.info("WALMART FORECASTING ETL PIPELINE")
        logger.info("="*60)
        
        # 1. Create Spark session
        logger.info("\n[STEP 1/4] Initializing Spark")
        spark = create_spark_session("WalmartForecastingETL")
        
        # 2. Extract data
        logger.info("\n[STEP 2/4] Extracting data from CSV files")
        train_df = read_training_data(spark)
        features_df = read_features_data(spark)
        stores_df = read_stores_data(spark)
        
        # Validate extracted data
        validate_data(train_df, "Training Data")
        validate_data(features_df, "Features Data")
        validate_data(stores_df, "Stores Data")
        
        # 3. Transform data
        logger.info("\n[STEP 3/4] Transforming data")
        processed_df = run_full_transformation(train_df, features_df, stores_df)
        
        # Show sample
        logger.info("\nSample of processed data:")
        processed_df.select(
            "store_id", "dept_id", "feature_date", 
            "weekly_sales", "sales_lag_1", "rolling_mean_4"
        ).show(5)
        
        # 4. Load data
        logger.info("\n[STEP 4/4] Loading processed data")
        load_processed_data(processed_df)
        
        # Success summary
        logger.info("\n" + "="*60)
        logger.info("✅ ETL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Total rows processed: {processed_df.count()}")
        logger.info(f"Total features: {len(processed_df.columns)}")
        logger.info("Data written to:")
        logger.info("  - PostgreSQL: engineered_features table")
        logger.info("  - Parquet: data/processed/features.parquet")
        
        # Stop Spark
        spark.stop()
        
    except Exception as e:
        logger.error(f"\n❌ ETL Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

### 5.6 Running the ETL Pipeline

**First time setup:**
```bash
# Make sure you're in the project root
cd walmart-forecasting

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Run the pipeline
python etl/run_pipeline.py
```

**Expected output:**
```
============================================================
WALMART FORECASTING ETL PIPELINE
============================================================

[STEP 1/4] Initializing Spark
✅ Spark session created: WalmartForecastingETL

[STEP 2/4] Extracting data from CSV files
Reading training data from data/raw/train.csv
✅ Training data loaded: 421570 rows
Reading features data from data/raw/features.csv
✅ Features data loaded: 6435 rows
Reading stores data from data/raw/stores.csv
✅ Stores data loaded: 45 rows

[STEP 3/4] Transforming data
Joining all tables
✅ Tables joined: 421570 rows
Creating time features
✅ Time features created
Creating lag features
✅ Lag features created
Creating rolling features
✅ Rolling features created
Creating economic features
✅ Economic features created
Creating markdown features
✅ Markdown features created
Creating store features
✅ Store features created
Cleaning column names
✅ Column names cleaned
Removing rows with null targets
✅ Final dataset: 415000 rows

[STEP 4/4] Loading processed data
Writing to PostgreSQL table: engineered_features
✅ Wrote 415000 rows to engineered_features
Writing to Parquet: data/processed/features.parquet
✅ Wrote Parquet file

============================================================
✅ ETL PIPELINE COMPLETED SUCCESSFULLY
============================================================
Total rows processed: 415000
Total features: 45
```

**This will take 5-15 minutes depending on your system.**

### 5.7 Verify ETL Output

**In pgAdmin4:**
```sql
-- Check row count
SELECT COUNT(*) FROM engineered_features;
-- Should return: ~415,000

-- View sample data
SELECT 
    store_id, 
    dept_id, 
    feature_date, 
    weekly_sales, 
    sales_lag_1, 
    rolling_mean_4,
    temperature,
    has_markdown
FROM engineered_features
ORDER BY feature_date DESC
LIMIT 10;

-- Check for nulls in key columns
SELECT 
    COUNT(*) FILTER (WHERE sales_lag_1 IS NULL) as null_lag_1,
    COUNT(*) FILTER (WHERE rolling_mean_4 IS NULL) as null_rolling_4,
    COUNT(*) FILTER (WHERE weekly_sales IS NULL) as null_target
FROM engineered_features;
```

---

## 6. Machine Learning Models

### 6.1 MLflow Setup

#### 6.1.1 Start MLflow Server

**Create `models/start_mlflow.sh`:**
```bash
#!/bin/bash
# Start MLflow tracking server

export MLFLOW_BACKEND_STORE_URI="postgresql://postgres:your_password@localhost:5432/mlflow_tracking"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="./data/models"

mlflow server \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port 5000
```

**Make executable and run:**
```bash
chmod +x models/start_mlflow.sh
./models/start_mlflow.sh
```

**Keep this terminal open. MLflow UI will be at http://localhost:5000**

#### 6.1.2 Test MLflow Connection

**Create `models/test_mlflow.py`:**
```python
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Create test experiment
experiment_name = "test_experiment"
mlflow.set_experiment(experiment_name)

# Log a test run
with mlflow.start_run(run_name="test_run"):
    mlflow.log_param("test_param", "value")
    mlflow.log_metric("test_metric", 0.95)
    
print("✅ MLflow connection successful!")
print(f"View UI at: {os.getenv('MLFLOW_TRACKING_URI')}")
```

**Run test:**
```bash
python models/test_mlflow.py
```

**Open http://localhost:5000 - you should see "test_experiment"**

### 6.2 Prophet Baseline Model

**Create `models/prophet_trainer.py`:**
```python
"""
Prophet baseline model for time series forecasting
"""
import pandas as pd
import mlflow
import mlflow.prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_from_postgres(store_id, dept_id=None):
    """
    Load historical sales data from PostgreSQL
    """
    engine = create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    
    if dept_id:
        query = f"""
        SELECT feature_date as ds, weekly_sales as y
        FROM engineered_features
        WHERE store_id = {store_id} AND dept_id = {dept_id}
        ORDER BY feature_date
        """
    else:
        query = f"""
        SELECT feature_date as ds, SUM(weekly_sales) as y
        FROM engineered_features
        WHERE store_id = {store_id}
        GROUP BY feature_date
        ORDER BY feature_date
        """
    
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} rows for Store {store_id}")
    
    return df

def train_prophet_model(df, store_id):
    """
    Train Prophet model with holidays and seasonality
    """
    logger.info(f"Training Prophet model for Store {store_id}")
    
    # Initialize Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )
    
    # Add US holidays
    model.add_country_holidays(country_name='US')
    
    # Fit model
    model.fit(df)
    
    logger.info("✅ Prophet model trained")
    return model

def evaluate_model(model, df):
    """
    Evaluate Prophet model using cross-validation
    """
    logger.info("Evaluating model with cross-validation")
    
    # Cross-validation: 365 days initial, 90 days period, 90 days horizon
    df_cv = cross_validation(
        model,
        initial='365 days',
        period='90 days',
        horizon='90 days',
        parallel="threads"
    )
    
    # Calculate performance metrics
    df_metrics = performance_metrics(df_cv)
    
    logger.info("Cross-validation metrics:")
    logger.info(f"MAPE: {df_metrics['mape'].mean():.4f}")
    logger.info(f"RMSE: {df_metrics['rmse'].mean():.2f}")
    logger.info(f"MAE: {df_metrics['mae'].mean():.2f}")
    
    return df_metrics

def forecast_future(model, periods=8):
    """
    Generate future predictions
    """
    logger.info(f"Generating forecast for {periods} weeks")
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq='W')
    
    # Generate forecast
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

def main():
    """
    Train Prophet models for all stores
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("prophet_baseline")
    
    # Train for top 5 stores by volume (for MVP)
    top_stores = [1, 2, 4, 6, 10]  # Adjust based on your data
    
    for store_id in top_stores:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Prophet for Store {store_id}")
        logger.info(f"{'='*60}")
        
        with mlflow.start_run(run_name=f"prophet_store_{store_id}"):
            # Log parameters
            mlflow.log_param("store_id", store_id)
            mlflow.log_param("model_type", "prophet")
            mlflow.log_param("seasonality_mode", "multiplicative")
            
            # Load data
            df = load_data_from_postgres(store_id)
            
            # Train model
            model = train_prophet_model(df, store_id)
            
            # Evaluate model
            metrics = evaluate_model(model, df)
            
            # Log metrics
            mlflow.log_metric("mape", metrics['mape'].mean())
            mlflow.log_metric("rmse", metrics['rmse'].mean())
            mlflow.log_metric("mae", metrics['mae'].mean())
            
            # Generate forecast
            forecast = forecast_future(model, periods=8)
            logger.info(f"\nForecast for next 8 weeks:")
            logger.info(forecast.to_string())
            
            # Log model
            mlflow.prophet.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"prophet_store_{store_id}"
            )
            
            logger.info(f"✅ Store {store_id} complete")
    
    logger.info("\n🎉 All Prophet models trained successfully!")
    logger.info(f"View results at: {os.getenv('MLFLOW_TRACKING_URI')}")

if __name__ == "__main__":
    main()
```

**Run Prophet training:**
```bash
python models/prophet_trainer.py
```

**This will take 10-20 minutes.**

### 6.3 LightGBM Model

**Create `models/lightgbm_trainer.py`:**
```python
"""
LightGBM model for retail demand forecasting
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """
    Load engineered features from PostgreSQL
    """
    engine = create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    
    query = """
    SELECT * FROM engineered_features
    ORDER BY store_id, dept_id, feature_date
    """
    
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} rows from database")
    
    return df

def prepare_features(df):
    """
    Select and prepare features for model training
    """
    # Features to use
    feature_cols = [
        # Time features
        'week_of_year', 'month', 'quarter', 'is_month_start', 'is_month_end',
        
        # Lag features
        'sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_lag_8', 'sales_lag_52',
        
        # Rolling features
        'rolling_mean_4', 'rolling_mean_13', 'rolling_mean_52',
        'rolling_std_4', 'rolling_std_13',
        'rolling_min_4', 'rolling_max_4',
        
        # Economic features
        'temperature', 'temperature_deviation',
        'fuel_price', 'fuel_price_change',
        'cpi', 'cpi_change',
        'unemployment', 'unemployment_change',
        
        # Markdown features
        'total_markdown', 'has_markdown', 'markdown_count',
        
        # Store features
        'store_type_A', 'store_type_B', 'store_type_C', 'size_normalized',
        
        # Holiday
        'is_holiday'
    ]
    
    # Target
    target_col = 'weekly_sales'
    
    # Remove rows with null values in key features
    df = df.dropna(subset=['sales_lag_1', 'rolling_mean_4', target_col])
    
    logger.info(f"Feature count: {len(feature_cols)}")
    logger.info(f"Rows after cleaning: {len(df)}")
    
    return df, feature_cols, target_col

def train_test_split_timeseries(df, test_size=0.2):
    """
    Time-based train/test split (no shuffle!)
    """
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    logger.info(f"Train set: {len(train_df)} rows")
    logger.info(f"Test set: {len(test_df)} rows")
    logger.info(f"Train date range: {train_df['feature_date'].min()} to {train_df['feature_date'].max()}")
    logger.info(f"Test date range: {test_df['feature_date'].min()} to {test_df['feature_date'].max()}")
    
    return train_df, test_df

def train_lightgbm(train_df, test_df, feature_cols, target_col):
    """
    Train LightGBM model
    """
    logger.info("Training LightGBM model")
    
    # Prepare datasets
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    logger.info("✅ LightGBM model trained")
    return model

def evaluate_model(model, test_df, feature_cols, target_col):
    """
    Evaluate model performance
    """
    logger.info("Evaluating model")
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Weighted MAE (competition metric)
    is_holiday = test_df['is_holiday'].values
    weights = np.where(is_holiday, 5, 1)
    wmae = np.sum(weights * np.abs(y_test - y_pred)) / np.sum(weights)
    
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"MAPE: {mape:.2f}%")
    logger.info(f"WMAE: {wmae:.2f} ⭐ (competition metric)")
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'wmae': wmae
    }
    
    return metrics, y_pred

def feature_importance_analysis(model, feature_cols):
    """
    Analyze feature importance
    """
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Most Important Features:")
    logger.info(feature_importance.head(10).to_string(index=False))
    
    return feature_importance

def main():
    """
    Complete model training pipeline
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("lightgbm_forecasting")
    mlflow.lightgbm.autolog()
    
    logger.info("="*60)
    logger.info("LIGHTGBM TRAINING PIPELINE")
    logger.info("="*60)
    
    with mlflow.start_run(run_name="lightgbm_global_model"):
        # 1. Load data
        logger.info("\n[1/5] Loading data")
        df = load_data()
        
        # 2. Prepare features
        logger.info("\n[2/5] Preparing features")
        df, feature_cols, target_col = prepare_features(df)
        
        # Log dataset info
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_samples", len(df))
        
        # 3. Train/test split
        logger.info("\n[3/5] Splitting data")
        train_df, test_df = train_test_split_timeseries(df)
        
        # 4. Train model
        logger.info("\n[4/5] Training model")
        model = train_lightgbm(train_df, test_df, feature_cols, target_col)
        
        # 5. Evaluate
        logger.info("\n[5/5] Evaluating model")
        metrics, predictions = evaluate_model(model, test_df, feature_cols, target_col)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Feature importance
        feature_importance = feature_importance_analysis(model, feature_cols)
        
        # Log feature importance as artifact
        importance_path = "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        
        # Register model
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name="lightgbm_forecaster"
        )
        
        logger.info("\n" + "="*60)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"WMAE: {metrics['wmae']:.2f}")
        logger.info(f"Target: < 700 (Baseline: 821)")
        logger.info(f"\nView results at: {os.getenv('MLFLOW_TRACKING_URI')}")

if __name__ == "__main__":
    main()
```

**Run LightGBM training:**
```bash
python models/lightgbm_trainer.py
```

**This will take 15-30 minutes.**

**Expected result: WMAE between 600-700 (beating the 821 baseline)**

### 6.4 Generate Forecasts

**Create `models/generate_forecasts.py`:**
```python
"""
Generate forecasts and write to database
"""
import pandas as pd
import mlflow
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import logging
from datetime import datetime, timedelta

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """
    Load production model from MLflow registry
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    model_name = "lightgbm_forecaster"
    model_stage = "Production"
    
    logger.info(f"Loading model: {model_name} ({model_stage})")
    
    model_uri = f"models:/{model_name}/{model_stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    
    logger.info("✅ Model loaded")
    return model

def prepare_forecast_features(engine, forecast_weeks=8):
    """
    Prepare features for forecasting
    """
    logger.info(f"Preparing features for {forecast_weeks} week forecast")
    
    # Get latest data
    query = """
    SELECT *
    FROM engineered_features
    ORDER BY store_id, dept_id, feature_date
    """
    
    df = pd.read_sql(query, engine)
    
    # Get last date
    last_date = df['feature_date'].max()
    logger.info(f"Last historical date: {last_date}")
    
    # Create future dates
    future_dates = pd.date_range(
        start=last_date + timedelta(weeks=1),
        periods=forecast_weeks,
        freq='W'
    )
    
    # For MVP: Use last known values as proxy for future
    # In production: Integrate with external APIs for real forecasts
    
    logger.info(f"Forecast dates: {future_dates[0]} to {future_dates[-1]}")
    
    return df, future_dates

def generate_predictions(model, features_df, future_dates):
    """
    Generate predictions for future dates
    """
    logger.info("Generating predictions")
    
    # Feature columns (must match training)
    feature_cols = [
        'week_of_year', 'month', 'quarter', 'is_month_start', 'is_month_end',
        'sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_lag_8', 'sales_lag_52',
        'rolling_mean_4', 'rolling_mean_13', 'rolling_mean_52',
        'rolling_std_4', 'rolling_std_13', 'rolling_min_4', 'rolling_max_4',
        'temperature', 'temperature_deviation', 'fuel_price', 'fuel_price_change',
        'cpi', 'cpi_change', 'unemployment', 'unemployment_change',
        'total_markdown', 'has_markdown', 'markdown_count',
        'store_type_A', 'store_type_B', 'store_type_C', 'size_normalized',
        'is_holiday'
    ]
    
    # Get latest features per store-dept
    latest_features = features_df.sort_values('feature_date').groupby(
        ['store_id', 'dept_id']
    ).tail(1)
    
    forecasts = []
    
    for _, row in latest_features.iterrows():
        store_id = row['store_id']
        dept_id = row['dept_id']
        
        for forecast_date in future_dates:
            # Prepare features (simplified for MVP)
            X = row[feature_cols].values.reshape(1, -1)
            
            # Predict
            pred = model.predict(X)[0]
            
            # Confidence interval (simplified: ±15%)
            lower = pred * 0.85
            upper = pred * 1.15
            
            forecasts.append({
                'store_id': int(store_id),
                'dept_id': int(dept_id),
                'forecast_date': forecast_date,
                'predicted_sales': float(pred),
                'prediction_lower': float(lower),
                'prediction_upper': float(upper),
                'model_name': 'lightgbm_forecaster',
                'model_version': '1',
                'confidence_score': 0.85
            })
    
    forecast_df = pd.DataFrame(forecasts)
    logger.info(f"✅ Generated {len(forecast_df)} predictions")
    
    return forecast_df

def write_forecasts_to_db(forecast_df, engine):
    """
    Write forecasts to database
    """
    logger.info("Writing forecasts to database")
    
    forecast_df.to_sql(
        'forecasts',
        engine,
        if_exists='append',
        index=False
    )
    
    logger.info("✅ Forecasts written to database")

def main():
    """
    Complete forecast generation pipeline
    """
    logger.info("="*60)
    logger.info("FORECAST GENERATION PIPELINE")
    logger.info("="*60)
    
    # Database connection
    engine = create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    
    # 1. Load model
    logger.info("\n[1/4] Loading model")
    model = load_model()
    
    # 2. Prepare features
    logger.info("\n[2/4] Preparing features")
    features_df, future_dates = prepare_forecast_features(engine)
    
    # 3. Generate predictions
    logger.info("\n[3/4] Generating predictions")
    forecast_df = generate_predictions(model, features_df, future_dates)
    
    # 4. Write to database
    logger.info("\n[4/4] Writing to database")
    write_forecasts_to_db(forecast_df, engine)
    
    logger.info("\n" + "="*60)
    logger.info("✅ FORECAST GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Forecasts generated: {len(forecast_df)}")
    logger.info(f"Date range: {future_dates[0]} to {future_dates[-1]}")

if __name__ == "__main__":
    main()
```

**Run forecast generation:**
```bash
python models/generate_forecasts.py
```

**Verify forecasts in database:**
```sql
SELECT COUNT(*) FROM forecasts;
-- Should see forecasts for 8 weeks ahead

SELECT 
    store_id, 
    dept_id, 
    forecast_date, 
    predicted_sales 
FROM forecasts 
ORDER BY forecast_date 
LIMIT 20;
```

---

## 7. Multi-Agent System

[Continuing with agents, dashboard, deployment, etc...]

**This documentation is getting very long. Would you like me to:**

1. **Continue with the remaining sections** (Agents, Dashboard, Deployment)?
2. **Create this as a separate file** to keep it manageable?
3. **Split into multiple markdown files** (one per major section)?

Let me know your preference!