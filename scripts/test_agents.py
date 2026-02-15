"""
Test Multi-Agent AI System
Demonstrates the capabilities of the AI agents
"""
import os
import sys
import pandas as pd
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.db_manager import db_manager
from agents.orchestrator import AgentOrchestrator
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data():
    """Load sample data for testing"""
    logger.info("Loading sample data from database...")
    
    engine = db_manager.connect()
    
    # Load recent historical sales
    historical_sales = pd.read_sql("""
        SELECT *
        FROM engineered_features
        WHERE feature_date >= '2012-09-01'
        ORDER BY feature_date DESC
        LIMIT 1000
    """, engine)
    
    logger.info(f"✓ Loaded {len(historical_sales)} historical sales records")
    
    # Create sample forecasts (using recent data as proxy)
    forecasts = historical_sales.copy()
    forecasts['forecast_date'] = forecasts['feature_date']
    forecasts['predicted_sales'] = forecasts['weekly_sales'] * 1.05  # 5% growth assumption
    forecasts = forecasts[['store_id', 'dept_id', 'forecast_date', 'predicted_sales']]
    
    logger.info(f"✓ Created {len(forecasts)} sample forecasts")
    
    return forecasts, historical_sales


def main():
    """Test the multi-agent system"""
    logger.info("="*60)
    logger.info("MULTI-AGENT AI SYSTEM TEST")
    logger.info("="*60)
    
    try:
        # Step 1: Load data
        logger.info("\n[1/3] Loading sample data...")
        forecasts, historical_sales = load_sample_data()
        
        # Step 2: Initialize orchestrator
        logger.info("\n[2/3] Initializing Agent Orchestrator...")
        orchestrator = AgentOrchestrator()
        
        # Step 3: Run comprehensive analysis
        logger.info("\n[3/3] Running comprehensive analysis...")
        logger.info("  (This may take 30-60 seconds as agents process the data)\n")
        
        # Test with Store 1, Department 1
        results = orchestrator.analyze_forecast(
            forecasts=forecasts,
            historical_sales=historical_sales,
            store_id=1,
            dept_id=1
        )
        
        # Display results
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS RESULTS")
        logger.info("="*60)
        
        print("\n" + results['summary'])
        
        print("\n" + "="*60)
        print("DETAILED INSIGHTS")
        print("="*60)
        
        print("\n[DEMAND FORECASTING AGENT]:")
        print("-" * 60)
        print(results['detailed_insights']['demand_analysis']['response'])
        
        print("\n\n[INVENTORY OPTIMIZATION AGENT]:")
        print("-" * 60)
        print(results['detailed_insights']['inventory_recommendations']['response'])
        
        print("\n\n[ANOMALY DETECTION AGENT]:")
        print("-" * 60)
        print(results['detailed_insights']['anomaly_detection']['response'])
        
        logger.info("\n" + "="*60)
        logger.info("✓ MULTI-AGENT SYSTEM TEST COMPLETE")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
