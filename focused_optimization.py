#!/usr/bin/env python3
"""
Focused Model Optimization Script
Tests key approaches to improve R² from 0.2504 to target 0.6+
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and preprocess data efficiently"""
    print("[DATA] Loading data...")
    
    # Load with UTF-8 encoding
    data = pd.read_csv("src/data/historical/present.selection.historic.csv", 
                       encoding='utf-8', dtype='str')
    
    # Basic cleaning
    for col in data.columns:
        data[col] = data[col].astype(str).str.strip('"').str.strip()
    data = data.fillna("NONE")
    
    # Lowercase categorical columns
    categorical_cols = ['employee_gender', 'product_target_gender', 
                       'product_utility_type', 'product_durability', 'product_type']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].str.lower()
    
    print(f"[OK] Loaded {len(data)} records")
    return data

def aggregate_data(data):
    """Aggregate selection events"""
    print("[AGGREGATE] Aggregating...")
    
    grouping_cols = [
        'employee_shop', 'employee_branch', 'employee_gender',
        'product_main_category', 'product_sub_category', 'product_brand',
        'product_color', 'product_durability', 'product_target_gender',
        'product_utility_type', 'product_type'
    ]
    
    # Aggregate by counting
    agg_data = data.groupby(grouping_cols).size().reset_index(name='selection_count')
    
    print(f"[OK] {len(data)} events -> {len(agg_data)} combinations")
    return agg_data, grouping_cols

def prepare_features(agg_data, grouping_cols):
    """Prepare features with label encoding"""
    print("[FEATURES] Preparing...")
    
    X = agg_data[grouping_cols].copy()
    y = agg_data['selection_count']
    
    # Label encode
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    print(f"[OK] Features: {X.shape}, Target stats: mean={y.mean():.2f}, std={y.std():.2f}")
    return X, y

def test_model(name, model, X, y):
    """Test a model configuration"""
    print(f"\n[TEST] {name}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = model.predict(X_val)
    r2_val = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Calculate overfitting
    overfitting = r2_val - cv_mean
    
    print(f"   Validation R2: {r2_val:.4f}")
    print(f"   CV R2: {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"   Overfitting: {overfitting:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   Time: {train_time:.1f}s")
    
    if overfitting > 0.15:
        print("   [WARNING] HIGH OVERFITTING")
    elif cv_mean > 0.3:
        print("   [GOOD] GOOD PERFORMANCE")
    
    return r2_val, cv_mean, overfitting

def main():
    """Main optimization workflow"""
    print("="*60)
    print("FOCUSED MODEL OPTIMIZATION")
    print("="*60)
    
    # Load and prepare data
    data = load_data()
    agg_data, grouping_cols = aggregate_data(data)
    X, y = prepare_features(agg_data, grouping_cols)
    
    results = []
    
    # Test 1: Our current best (Conservative XGBoost)
    print("\n" + "="*40)
    print("PHASE 1: XGBoost Variants")
    print("="*40)
    
    xgb_conservative = XGBRegressor(
        n_estimators=1500, max_depth=6, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.5, reg_lambda=0.5,
        gamma=0.2, min_child_weight=10, random_state=42, n_jobs=-1
    )
    r2_val, cv_mean, overfitting = test_model("XGB Conservative", xgb_conservative, X, y)
    results.append(("XGB Conservative", r2_val, cv_mean, overfitting))
    
    # Test 2: Even more conservative (reduce overfitting)
    xgb_ultra_conservative = XGBRegressor(
        n_estimators=1000, max_depth=4, learning_rate=0.01,
        subsample=0.95, colsample_bytree=0.95, reg_alpha=1.0, reg_lambda=1.0,
        gamma=0.5, min_child_weight=20, random_state=42, n_jobs=-1
    )
    r2_val, cv_mean, overfitting = test_model("XGB Ultra Conservative", xgb_ultra_conservative, X, y)
    results.append(("XGB Ultra Conservative", r2_val, cv_mean, overfitting))
    
    # Test 3: Random Forest (naturally less prone to overfitting)
    print("\n" + "="*40)
    print("PHASE 2: Alternative Algorithms")
    print("="*40)
    
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=10, min_samples_split=20,
        min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1
    )
    r2_val, cv_mean, overfitting = test_model("Random Forest", rf, X, y)
    results.append(("Random Forest", r2_val, cv_mean, overfitting))
    
    # Test 4: More conservative Random Forest
    rf_conservative = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_split=50,
        min_samples_leaf=20, max_features='sqrt', random_state=42, n_jobs=-1
    )
    r2_val, cv_mean, overfitting = test_model("RF Conservative", rf_conservative, X, y)
    results.append(("RF Conservative", r2_val, cv_mean, overfitting))
    
    # Test 5: Log-transformed target
    print("\n" + "="*40)
    print("PHASE 3: Target Transformation")
    print("="*40)
    
    y_log = np.log1p(y)  # log(1 + y) to handle zeros
    
    xgb_log = XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.3, reg_lambda=0.3,
        gamma=0.1, min_child_weight=10, random_state=42, n_jobs=-1
    )
    r2_val, cv_mean, overfitting = test_model("XGB Log Target", xgb_log, X, y_log)
    results.append(("XGB Log Target", r2_val, cv_mean, overfitting))
    
    # Test 6: Feature selection (top important features only)
    print("\n" + "="*40)
    print("PHASE 4: Feature Analysis")
    print("="*40)
    
    # Train a simple model to get feature importance
    temp_model = XGBRegressor(n_estimators=100, random_state=42)
    temp_model.fit(X, y)
    
    # Get top 6 features (roughly half)
    feature_importance = temp_model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-6:]
    X_reduced = X.iloc[:, top_features_idx]
    
    print(f"[INFO] Using top 6 features (from {X.shape[1]})")
    
    xgb_reduced = XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.3, reg_lambda=0.3,
        gamma=0.1, min_child_weight=8, random_state=42, n_jobs=-1
    )
    r2_val, cv_mean, overfitting = test_model("XGB Reduced Features", xgb_reduced, X_reduced, y)
    results.append(("XGB Reduced Features", r2_val, cv_mean, overfitting))
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    # Sort by CV R² (more reliable metric)
    results.sort(key=lambda x: x[2], reverse=True)
    
    print("Ranking by Cross-Validation R² (most reliable):")
    for i, (name, val_r2, cv_r2, overfit) in enumerate(results):
        status = "[WARNING] HIGH OVERFIT" if overfit > 0.15 else "[STABLE]" if overfit < 0.1 else "[MODERATE]"
        print(f"  {i+1}. {name:20} CV R²: {cv_r2:.4f}, Val R²: {val_r2:.4f} {status}")
    
    best_stable = min(results, key=lambda x: x[3])  # Least overfitting
    best_cv = max(results, key=lambda x: x[2])      # Best CV performance
    
    print(f"\nBest stable model: {best_stable[0]} (overfitting: {best_stable[3]:.4f})")
    print(f"Best CV performance: {best_cv[0]} (CV R²: {best_cv[2]:.4f})")
    
    if best_cv[2] >= 0.6:
        print("\n[SUCCESS] TARGET ACHIEVED: CV R2 >= 0.6")
    elif best_cv[2] >= 0.4:
        print(f"\n[MODERATE] CV R2 = {best_cv[2]:.4f} (target: 0.6)")
    else:
        print(f"\n[POOR] CV R2 = {best_cv[2]:.4f} (may need data strategy review)")
    
    print("="*60)
    return best_cv[2]

if __name__ == "__main__":
    best_cv_r2 = main()
    print(f"\nFinal Best CV R²: {best_cv_r2:.4f}")