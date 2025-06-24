"""
Validation script to check model artifacts and ensure production readiness.
This script validates the current model state and provides migration guidance.
"""

import os
import pickle
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_model_artifacts():
    """Check current model artifacts and validate production readiness."""
    
    print("=== Model Artifacts Validation ===\n")
    
    # Check both possible model directories
    base_dir = Path(__file__).parent.parent
    old_model_dir = base_dir / "models" / "catboost_rmse_model"
    new_model_dir = base_dir / "models" / "catboost_poisson_model"
    
    print(f"Checking old model directory: {old_model_dir}")
    print(f"Checking new model directory: {new_model_dir}")
    
    old_exists = old_model_dir.exists()
    new_exists = new_model_dir.exists()
    
    print(f"\nOld directory exists: {old_exists}")
    print(f"New directory exists: {new_exists}")
    
    # Determine migration status
    if old_exists and not new_exists:
        print("\n🔄 MIGRATION NEEDED:")
        print("- Old 'catboost_rmse_model' directory found")
        print("- New 'catboost_poisson_model' directory not found")
        print("- Recommendation: Rename directory for consistency")
        
        migrate_model_artifacts(old_model_dir, new_model_dir)
        
    elif new_exists:
        print("\n✅ PRODUCTION READY:")
        print("- New 'catboost_poisson_model' directory found")
        validate_production_artifacts(new_model_dir)
        
    elif old_exists and new_exists:
        print("\n⚠️ BOTH DIRECTORIES EXIST:")
        print("- Both old and new directories found")
        print("- Recommendation: Remove old directory after validation")
        validate_production_artifacts(new_model_dir)
        
    else:
        print("\n❌ NO MODEL FOUND:")
        print("- No trained model artifacts found")
        print("- Run training pipeline first: python src/ml/catboost_trainer.py")

def migrate_model_artifacts(old_dir: Path, new_dir: Path):
    """Migrate model artifacts from old to new directory naming."""
    
    print(f"\nMigrating artifacts from {old_dir} to {new_dir}...")
    
    try:
        # Create new directory
        new_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all files
        import shutil
        for item in old_dir.iterdir():
            if item.is_file():
                old_name = item.name
                # Rename model file if it contains 'rmse'
                if 'rmse' in old_name and old_name.endswith('.cbm'):
                    new_name = old_name.replace('rmse', 'poisson')
                else:
                    new_name = old_name
                
                shutil.copy2(item, new_dir / new_name)
                print(f"  Copied: {old_name} -> {new_name}")
        
        print(f"\n✅ Migration completed successfully!")
        print(f"New artifacts location: {new_dir}")
        
        # Validate the migrated artifacts
        validate_production_artifacts(new_dir)
        
        print(f"\n📝 NEXT STEPS:")
        print(f"1. Verify the migrated model works correctly")
        print(f"2. Remove old directory: {old_dir}")
        print(f"3. Update any hardcoded paths in scripts")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")

def validate_production_artifacts(model_dir: Path):
    """Validate that all required production artifacts exist and are valid."""
    
    print(f"\nValidating production artifacts in {model_dir}...")
    
    required_files = [
        "catboost_poisson_model.cbm",
        "model_metadata.pkl",
        "model_params.pkl",
        "model_metrics.pkl",
        "historical_features_snapshot.pkl",
        "branch_mapping_snapshot.pkl",
        "global_defaults_snapshot.pkl"
    ]
    
    optional_files = [
        "product_relativity_features.csv"
    ]
    
    # Check required files
    missing_files = []
    for file_name in required_files:
        file_path = model_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} (MISSING)")
            missing_files.append(file_name)
    
    # Check optional files
    for file_name in optional_files:
        file_path = model_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name} (optional)")
        else:
            print(f"  ⚠️ {file_name} (optional, not found)")
    
    if missing_files:
        print(f"\n❌ VALIDATION FAILED:")
        print(f"Missing required files: {missing_files}")
        return False
    
    # Validate metadata content
    try:
        metadata_path = model_dir / "model_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"\n📊 MODEL METADATA VALIDATION:")
        
        # Check model type
        model_type = metadata.get('model_type', 'Unknown')
        print(f"  Model Type: {model_type}")
        
        # Check performance metrics
        performance_metrics = metadata.get('performance_metrics', {})
        
        print(f"  Performance Metrics:")
        for metric_name, value in performance_metrics.items():
            if isinstance(value, (int, float)):
                print(f"    {metric_name}: {value:.4f}")
            else:
                print(f"    {metric_name}: {value}")
        
        # Validate Poisson-specific metrics
        poisson_metrics = ['poisson_deviance', 'business_mape']
        missing_poisson_metrics = [m for m in poisson_metrics if m not in performance_metrics]
        
        if missing_poisson_metrics:
            print(f"  ⚠️ Missing Poisson metrics: {missing_poisson_metrics}")
        else:
            print(f"  ✅ All Poisson metrics present")
        
        # Check feature information
        features_used = metadata.get('features_used', [])
        categorical_features = metadata.get('categorical_features_in_model', [])
        numeric_medians = metadata.get('numeric_feature_medians', {})
        
        print(f"  Features: {len(features_used)} total")
        print(f"  Categorical: {len(categorical_features)}")
        print(f"  Numeric medians: {len(numeric_medians)}")
        
        # Check for interaction hash features
        hash_features = [f for f in features_used if f.startswith('interaction_hash_')]
        print(f"  Interaction hash features: {len(hash_features)}")
        
        if len(hash_features) != 96:
            print(f"  ⚠️ Expected 96 interaction hash features, found {len(hash_features)}")
        else:
            print(f"  ✅ Correct number of interaction hash features")
        
        print(f"\n✅ PRODUCTION VALIDATION PASSED")
        
        # Print summary for ops team
        print(f"\n📋 VALIDATION SNAPSHOT FOR OPS:")
        if 'poisson_deviance' in performance_metrics:
            print(f"  Validation Poisson deviance: {performance_metrics['poisson_deviance']:.4f}")
        if 'business_mape' in performance_metrics:
            print(f"  Weighted Business MAPE: {performance_metrics['business_mape']:.2f}%")
        if 'r2_validation' in performance_metrics:
            print(f"  R² (count space): {performance_metrics['r2_validation']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ METADATA VALIDATION FAILED: {e}")
        return False

def test_predictor_loading():
    """Test that the predictor can load successfully."""
    
    print(f"\n🧪 TESTING PREDICTOR LOADING...")
    
    try:
        from src.ml.predictor import get_predictor
        
        # Try to load predictor
        predictor = get_predictor()
        
        print(f"  ✅ Predictor loaded successfully")
        print(f"  Model path: {predictor.model_path}")
        print(f"  Expected columns: {len(predictor.expected_columns)}")
        print(f"  Categorical features: {len(predictor.categorical_features)}")
        
        if predictor.model_rmse is not None:
            print(f"  Model RMSE: {predictor.model_rmse:.4f}")
        if predictor.model_poisson is not None:
            print(f"  Model Poisson: {predictor.model_poisson:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Predictor loading failed: {e}")
        return False

if __name__ == "__main__":
    check_model_artifacts()
    test_predictor_loading()
    
    print(f"\n=== VALIDATION COMPLETE ===")