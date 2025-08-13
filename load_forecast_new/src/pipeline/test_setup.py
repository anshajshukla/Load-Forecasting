"""
Test script to validate the data pipeline setup.
Run this script to ensure all components are working correctly.
"""

import sys
import os
from pathlib import Path
import datetime
import json

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all required modules can be imported."""
    print("🔧 Testing imports...")
    
    try:
        import numpy as np
        print("  ✅ NumPy imported successfully")
    except ImportError as e:
        print(f"  ❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("  ✅ Pandas imported successfully")
    except ImportError as e:
        print(f"  ❌ Pandas import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"  ❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import requests
        print("  ✅ Requests imported successfully")
    except ImportError as e:
        print(f"  ❌ Requests import failed: {e}")
        return False
    
    try:
        from bs4 import BeautifulSoup
        print("  ✅ BeautifulSoup imported successfully")
    except ImportError as e:
        print(f"  ❌ BeautifulSoup import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"  ✅ Scikit-learn {sklearn.__version__} imported successfully")
    except ImportError as e:
        print(f"  ❌ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_database_manager():
    """Test database manager functionality."""
    print("\n🗄️  Testing database manager...")
    
    try:
        from database.db_manager import DatabaseManager, create_database_manager
        
        # Create database manager
        db_manager = create_database_manager()
        print("  ✅ Database manager created successfully")
        
        # Test table creation
        db_manager.create_tables()
        print("  ✅ Database tables created successfully")
        
        # Test simple query
        result = db_manager.execute_query("SELECT 1 as test")
        if not result.empty and result['test'].iloc[0] == 1:
            print("  ✅ Database query test passed")
        else:
            print("  ❌ Database query test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Database manager test failed: {e}")
        return False

def test_data_fetcher():
    """Test data fetcher functionality."""
    print("\n📡 Testing data fetcher...")
    
    try:
        from enhanced_fetcher import EnhancedDataFetcher
        
        # Create data fetcher
        fetcher = EnhancedDataFetcher()
        print("  ✅ Data fetcher created successfully")
        
        # Test network connectivity
        import requests
        response = requests.get("https://www.delhisldc.org", timeout=10)
        if response.status_code == 200:
            print("  ✅ Network connectivity to Delhi SLDC confirmed")
        else:
            print(f"  ⚠️  Delhi SLDC website returned status {response.status_code}")
        
        # Test with a recent date
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        date_str = yesterday.strftime('%d/%m/%Y')
        
        print(f"  🔍 Testing data fetch for {date_str}...")
        df = fetcher.fetch_daily_data(date_str, include_weather=False)
        
        if df is not None and not df.empty:
            print(f"  ✅ Data fetch successful! Retrieved {len(df)} records")
            print(f"  📊 Sample data: DELHI={df['DELHI'].iloc[0]:.1f} MW")
        else:
            print("  ⚠️  Data fetch returned empty result (this might be normal for recent dates)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data fetcher test failed: {e}")
        return False

def test_training_pipeline():
    """Test training pipeline setup."""
    print("\n🧠 Testing training pipeline...")
    
    try:
        from training_pipeline import ModelTrainingPipeline, TrainingConfig
        
        # Create training configuration
        config = TrainingConfig(
            sequence_length=24,
            prediction_horizon=24,
            epochs=1,  # Just 1 epoch for testing
            batch_size=32
        )
        
        # Create training pipeline
        pipeline = ModelTrainingPipeline(config=config)
        print("  ✅ Training pipeline created successfully")
        
        # Test model building
        input_shape = (24, 20)  # 24 time steps, 20 features
        output_shape = 24  # 24-hour prediction
        
        model = pipeline.build_model(input_shape, output_shape)
        print(f"  ✅ Model architecture built successfully")
        print(f"  📋 Model parameters: {model.count_params():,}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training pipeline test failed: {e}")
        return False

def test_validation_pipeline():
    """Test validation pipeline setup."""
    print("\n✅ Testing validation pipeline...")
    
    try:
        from validation_pipeline import ValidationPipeline, ValidationConfig
        
        # Create validation configuration
        config = ValidationConfig(
            validation_window=24,
            min_accuracy_threshold=85.0
        )
        
        # Create validation pipeline
        pipeline = ValidationPipeline(config=config)
        print("  ✅ Validation pipeline created successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Validation pipeline test failed: {e}")
        return False

def test_configuration():
    """Test configuration file loading."""
    print("\n⚙️  Testing configuration...")
    
    try:
        config_path = Path("config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("  ✅ Configuration file loaded successfully")
            
            # Validate key sections
            required_sections = ['pipeline', 'data_collection', 'training', 'validation', 'database']
            for section in required_sections:
                if section in config:
                    print(f"  ✅ {section} configuration found")
                else:
                    print(f"  ❌ {section} configuration missing")
                    return False
            
            return True
        else:
            print("  ❌ Configuration file not found")
            return False
            
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure creation."""
    print("\n📁 Testing directory structure...")
    
    try:
        # Expected directories
        expected_dirs = [
            "data_pipeline/database",
            "data_pipeline/models",
            "data_pipeline/validation",
            "data_pipeline/raw_data",
            "data_pipeline/logs"
        ]
        
        for dir_path in expected_dirs:
            path = Path(dir_path)
            if path.exists():
                print(f"  ✅ {dir_path} exists")
            else:
                print(f"  📁 Creating {dir_path}...")
                path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ {dir_path} created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Directory structure test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Delhi SLDC Load Forecasting Pipeline - Setup Validation")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Directory Structure", test_directory_structure),
        ("Configuration", test_configuration),
        ("Database Manager", test_database_manager),
        ("Data Fetcher", test_data_fetcher),
        ("Training Pipeline", test_training_pipeline),
        ("Validation Pipeline", test_validation_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:10} | {test_name}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Your pipeline is ready to run.")
        print("\nNext steps:")
        print("1. Run: python main_orchestrator.py --action start")
        print("2. Monitor: python main_orchestrator.py --action status")
        print("3. Check logs: data_pipeline/logs/pipeline.log")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please fix the issues before running the pipeline.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install tensorflow pandas numpy scikit-learn beautifulsoup4 requests matplotlib seaborn schedule joblib")
        print("- Check network connectivity")
        print("- Verify file permissions")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
