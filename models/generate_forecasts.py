"""
Generate Forecasts and Write to Database
Loads trained LightGBM model and generates 8-week predictions
"""
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
from datetime import timedelta
from pathlib import Path
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_db_engine():
    """Create database engine with URL-encoded password"""
    password = quote_plus(os.getenv('DB_PASSWORD'))
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{password}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )


def load_model():
    """
    Load trained LightGBM model
    Tries local file first, then MLflow
    """
    local_path = 'models/saved/walmart_forecaster.txt'

    # Option 1: Load from local file
    if Path(local_path).exists():
        model = lgb.Booster(model_file=local_path)
        logger.info(f"✅ Model loaded from local file: {local_path}")
        return model

    # Option 2: Load from MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

        # Try the name used in trainer.py
        model_name = "walmart_sales_forecaster"
        logger.info(f"Local model not found. Trying MLflow: {model_name}")

        # Get latest version
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name)

        if versions:
            model_uri = f"models:/{model_name}/{versions[0].version}"
            model = mlflow.lightgbm.load_model(model_uri)
            logger.info(f"✅ Model loaded from MLflow: {model_uri}")
            return model
        else:
            raise FileNotFoundError(f"No versions found for {model_name}")

    except Exception as e:
        logger.error(f"❌ Could not load model from MLflow: {e}")

    raise FileNotFoundError(
        "No trained model found!\n"
        "Please train the model first by running:\n"
        "  python models/train.py\n"
        f"Expected local file at: {local_path}"
    )


def load_latest_features(engine):
    """
    Load the most recent features for each store-dept combination
    These serve as the basis for generating future predictions
    """
    logger.info("Loading latest features from database...")

    query = """
        SELECT *
        FROM engineered_features ef
        WHERE feature_date = (
            SELECT MAX(feature_date)
            FROM engineered_features ef2
            WHERE ef2.store_id = ef.store_id
              AND ef2.dept_id = ef.dept_id
        )
        ORDER BY store_id, dept_id
    """

    df = pd.read_sql(query, engine)
    logger.info(f"✅ Loaded features for {len(df)} store-dept combinations")
    logger.info(f"   Latest date in data: {df['feature_date'].max()}")

    return df


def get_feature_columns(model):
    """
    Get the feature columns the model expects
    """
    # LightGBM Booster stores feature names
    if hasattr(model, 'feature_name'):
        features = model.feature_name()
        logger.info(f"✅ Model expects {len(features)} features")
        return features

    # Fallback: define manually (must match training)
    logger.warning("Could not read feature names from model. Using default list.")
    return [
        'week_of_year', 'month', 'quarter', 'year',
        'day_of_week', 'day_of_month', 'day_of_year',
        'is_month_start', 'is_month_end',
        'is_quarter_start', 'is_quarter_end',
        'sales_lag_1', 'sales_lag_2', 'sales_lag_4',
        'sales_lag_8', 'sales_lag_52',
        'rolling_mean_4', 'rolling_mean_13', 'rolling_mean_52',
        'rolling_std_4', 'rolling_std_13',
        'rolling_min_4', 'rolling_max_4',
        'temperature', 'temperature_deviation',
        'fuel_price', 'fuel_price_change',
        'cpi', 'cpi_change',
        'unemployment', 'unemployment_change',
        'total_markdown', 'has_markdown', 'markdown_count',
        'store_type_A', 'store_type_B', 'store_type_C',
        'size_normalized', 'is_holiday'
    ]


def generate_predictions(model, features_df, forecast_weeks=8):
    """
    Generate predictions for future weeks

    For each store-dept, uses the latest known features as a base
    and generates predictions for the next N weeks.
    """
    logger.info(f"Generating {forecast_weeks}-week forecasts...")

    feature_cols = get_feature_columns(model)
    last_date = pd.to_datetime(features_df['feature_date'].max())

    # Generate future dates (weekly)
    future_dates = [last_date + timedelta(weeks=w) for w in range(1, forecast_weeks + 1)]
    logger.info(f"   Forecast dates: {future_dates[0].date()} to {future_dates[-1].date()}")

    all_forecasts = []

    for _, row in features_df.iterrows():
        store_id = int(row['store_id'])
        dept_id = int(row['dept_id'])

        for forecast_date in future_dates:
            # Build feature vector from latest known data
            # Update time features for the forecast date
            feature_row = row.copy()
            feature_row['week_of_year'] = forecast_date.isocalendar()[1]
            feature_row['month'] = forecast_date.month
            feature_row['quarter'] = (forecast_date.month - 1) // 3 + 1

            if 'year' in feature_cols:
                feature_row['year'] = forecast_date.year
            if 'day_of_week' in feature_cols:
                feature_row['day_of_week'] = forecast_date.weekday()
            if 'day_of_month' in feature_cols:
                feature_row['day_of_month'] = forecast_date.day
            if 'day_of_year' in feature_cols:
                feature_row['day_of_year'] = forecast_date.timetuple().tm_yday

            # Update period flags
            feature_row['is_month_start'] = 1 if forecast_date.day <= 7 else 0
            feature_row['is_month_end'] = 1 if forecast_date.day >= 24 else 0
            feature_row['is_quarter_start'] = 1 if forecast_date.month in [1, 4, 7, 10] else 0
            feature_row['is_quarter_end'] = 1 if forecast_date.month in [3, 6, 9, 12] else 0

            # Extract features in the correct order
            available = [c for c in feature_cols if c in feature_row.index]
            missing = [c for c in feature_cols if c not in feature_row.index]

            if missing:
                logger.warning(f"Missing features (will be 0): {missing[:5]}...")

            X = pd.DataFrame([feature_row[available]])

            # Add missing columns as 0
            for col in missing:
                X[col] = 0

            X = X[feature_cols]  # Ensure correct column order

            # Predict
            pred = model.predict(X)[0]
            pred = max(pred, 0)  # Sales can't be negative

            # Confidence interval (based on prediction magnitude)
            # Wider intervals for larger predictions
            std_estimate = max(pred * 0.15, 100)  # At least $100 uncertainty
            lower = max(pred - 1.96 * std_estimate, 0)
            upper = pred + 1.96 * std_estimate

            all_forecasts.append({
                'store_id': store_id,
                'dept_id': dept_id,
                'forecast_date': forecast_date.date(),
                'predicted_sales': round(float(pred), 2),
                'prediction_lower': round(float(lower), 2),
                'prediction_upper': round(float(upper), 2),
                'model_name': 'lightgbm_forecaster',
                'model_version': 'v1.0',
                'confidence_score': round(0.85 - (0.02 * future_dates.index(forecast_date)), 4)
            })

    forecast_df = pd.DataFrame(all_forecasts)
    logger.info(f"✅ Generated {len(forecast_df)} predictions")
    logger.info(f"   Stores: {forecast_df['store_id'].nunique()}")
    logger.info(f"   Departments: {forecast_df['dept_id'].nunique()}")
    logger.info(f"   Avg predicted weekly sales: ${forecast_df['predicted_sales'].mean():,.2f}")

    return forecast_df


def write_forecasts(forecast_df, engine):
    """Write forecasts to database, replacing any existing ones"""
    logger.info("Writing forecasts to database...")

    # Clear old forecasts
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM forecasts"))
        conn.commit()
        logger.info("   Cleared old forecasts")

    # Write new forecasts
    forecast_df.to_sql(
        'forecasts',
        engine,
        if_exists='append',
        index=False
    )

    logger.info(f"✅ Wrote {len(forecast_df)} forecasts to database")


def main():
    logger.info("=" * 60)
    logger.info("FORECAST GENERATION PIPELINE")
    logger.info("=" * 60)

    # 1. Database connection
    logger.info("\n[1/4] Connecting to database...")
    engine = get_db_engine()

    # 2. Load model
    logger.info("\n[2/4] Loading trained model...")
    model = load_model()

    # 3. Load features and generate predictions
    logger.info("\n[3/4] Loading features and generating predictions...")
    features_df = load_latest_features(engine)

    if features_df.empty:
        logger.error("❌ No features found! Run the ETL pipeline first.")
        return

    forecast_df = generate_predictions(model, features_df, forecast_weeks=8)

    # 4. Write to database
    logger.info("\n[4/4] Writing forecasts to database...")
    write_forecasts(forecast_df, engine)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ FORECAST GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total forecasts: {len(forecast_df)}")
    logger.info(f"Date range: {forecast_df['forecast_date'].min()} to {forecast_df['forecast_date'].max()}")
    logger.info(f"Avg predicted sales: ${forecast_df['predicted_sales'].mean():,.2f}")
    logger.info("\nVerify in pgAdmin:")
    logger.info("  SELECT COUNT(*) FROM forecasts;")
    logger.info("  SELECT * FROM forecasts LIMIT 10;")


if __name__ == "__main__":
    main()