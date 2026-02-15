"""
LightGBM Model Trainer with MLflow Tracking
Handles model training, evaluation, and persistence
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from datetime import datetime
import logging
import yaml
from pathlib import Path

from models.metrics import calculate_wmae, evaluate_model, wmae_lgb_metric
from database.db_manager import db_manager

logger = logging.getLogger(__name__)


class WalmartForecaster:
    """
    LightGBM-based forecasting model for Walmart retail sales
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize forecaster with configuration
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.mlflow_config = self.config['mlflow']
        
        # Model attributes
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
        # MLflow setup
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        mlflow.set_experiment(self.mlflow_config['experiment_name'])
        
        logger.info("✓ WalmartForecaster initialized")
        logger.info(f"  MLflow experiment: {self.mlflow_config['experiment_name']}")
    
    def prepare_data(self, df, target_col='weekly_sales', exclude_cols=None):
        """
        Prepare features and target from dataframe
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            exclude_cols: List of columns to exclude from features
        
        Returns:
            tuple: (X, y, is_holiday)
        """
        if exclude_cols is None:
            exclude_cols = ['id', 'store_id', 'dept_id', 'feature_date', 
                          'weekly_sales', 'created_at']
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].values
        is_holiday = df['is_holiday'].values if 'is_holiday' in df.columns else np.zeros(len(df), dtype=bool)
        
        # Store feature names
        self.feature_names = feature_cols
        
        logger.info(f"✓ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, is_holiday
    
    def create_train_val_split(self, df, val_weeks=8):
        """
        Create train/validation split based on time
        
        Args:
            df: DataFrame with feature_date column
            val_weeks: Number of weeks for validation
        
        Returns:
            tuple: (train_df, val_df)
        """
        df = df.sort_values('feature_date')
        
        # Get the last date
        max_date = df['feature_date'].max()
        
        # Calculate validation start date (val_weeks before max_date)
        val_start_date = max_date - pd.Timedelta(weeks=val_weeks)
        
        # Split data
        train_df = df[df['feature_date'] < val_start_date].copy()
        val_df = df[df['feature_date'] >= val_start_date].copy()
        
        logger.info(f"✓ Train/Val split created:")
        logger.info(f"  Train: {len(train_df):,} samples ({train_df['feature_date'].min()} to {train_df['feature_date'].max()})")
        logger.info(f"  Val:   {len(val_df):,} samples ({val_df['feature_date'].min()} to {val_df['feature_date'].max()})")
        
        return train_df, val_df
    
    def train(self, train_df, val_df=None):
        """
        Train LightGBM model with MLflow tracking
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe (optional)
        
        Returns:
            dict: Training metrics
        """
        logger.info("="*60)
        logger.info("TRAINING LIGHTGBM MODEL")
        logger.info("="*60)
        
        # Prepare data
        X_train, y_train, is_holiday_train = self.prepare_data(train_df)
        
        if val_df is not None:
            X_val, y_val, is_holiday_val = self.prepare_data(val_df)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if val_df is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.model_config['params'])
            mlflow.log_param('train_samples', len(X_train))
            if val_df is not None:
                mlflow.log_param('val_samples', len(X_val))
            
            # Train model
            logger.info("Training model...")
            self.model = lgb.train(
                self.model_config['params'],
                train_data,
                num_boost_round=self.model_config['num_boost_round'],
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=self.model_config['early_stopping_rounds']),
                    lgb.log_evaluation(period=50)
                ]
            )
            
            logger.info(f"✓ Training complete. Best iteration: {self.model.best_iteration}")
            
            # Evaluate on training set
            train_pred = self.model.predict(X_train)
            train_metrics = evaluate_model(y_train, train_pred, is_holiday_train)
            
            # Log training metrics
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f'train_{metric_name}', metric_value)
            
            # Evaluate on validation set
            if val_df is not None:
                val_pred = self.model.predict(X_val)
                val_metrics = evaluate_model(y_val, val_pred, is_holiday_val)
                
                # Log validation metrics
                for metric_name, metric_value in val_metrics.items():
                    mlflow.log_metric(f'val_{metric_name}', metric_value)
            
            # Feature importance
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            
            # Log top 20 features
            logger.info("\nTop 20 Most Important Features:")
            for idx, row in self.feature_importance.head(20).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.2f}")
            
            # Save feature importance as artifact
            importance_path = 'feature_importance.csv'
            self.feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            os.remove(importance_path)
            
            # Log model
            mlflow.lightgbm.log_model(
                self.model,
                artifact_path='model',
                registered_model_name='walmart_sales_forecaster'
            )
            
            # Get run info
            run_id = mlflow.active_run().info.run_id
            logger.info(f"\n✓ MLflow run ID: {run_id}")
            
            # Save to database
            self._save_model_metadata(
                run_id=run_id,
                wmae=val_metrics['wmae'] if val_df is not None else train_metrics['wmae'],
                mae=val_metrics['mae'] if val_df is not None else train_metrics['mae'],
                rmse=val_metrics['rmse'] if val_df is not None else train_metrics['rmse']
            )
            
            return val_metrics if val_df is not None else train_metrics
    
    def _save_model_metadata(self, run_id, wmae, mae, rmse):
        """
        Save model metadata to database
        
        Args:
            run_id: MLflow run ID
            wmae: WMAE score
            mae: MAE score
            rmse: RMSE score
        """
        import json
        
        engine = db_manager.connect()
        
        metadata = pd.DataFrame([{
            'run_id': run_id,
            'model_name': 'lightgbm_forecaster',
            'model_version': 'v1.0',
            'wmae': wmae,
            'mae': mae,
            'rmse': rmse,
            'training_date': datetime.now(),
            'parameters': json.dumps(self.model_config['params']),
            'feature_importance': json.dumps(self.feature_importance.to_dict('records'))
        }])
        
        metadata.to_sql('model_metadata', engine, if_exists='append', index=False)
        logger.info("✓ Model metadata saved to database")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature dataframe
        
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def save_model(self, path='models/saved/walmart_forecaster.txt'):
        """
        Save model to file
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(path)
        logger.info(f"✓ Model saved to {path}")
    
    def load_model(self, path='models/saved/walmart_forecaster.txt'):
        """
        Load model from file
        
        Args:
            path: Path to model file
        """
        self.model = lgb.Booster(model_file=path)
        logger.info(f"✓ Model loaded from {path}")
