#!/usr/bin/env python3
"""
Force reload of the cached predictor instance to pick up the latest trained model.
This is needed after retraining the model to ensure production uses the new model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.predictor import get_predictor
import ml.predictor as predictor_module

def force_reload_predictor():
    """Force reload the predictor by clearing the cached instance."""
    print("=== Forcing Model Reload ===")
    
    # Clear the cached instance
    predictor_module._predictor_instance = None
    print("Cleared cached predictor instance.")
    
    # Force creation of new instance with latest model
    print("Loading fresh predictor instance...")
    new_predictor = get_predictor()
    
    print("✅ Fresh predictor loaded successfully!")
    print(f"Model path: {new_predictor.model_path}")
    print(f"Loaded features: {len(new_predictor.expected_columns)}")
    print(f"Model loaded at: {new_predictor.model}")
    
    return new_predictor

if __name__ == "__main__":
    force_reload_predictor()
    print("Model reload complete. Production API should now use the updated model.")