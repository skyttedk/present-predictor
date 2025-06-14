#!/usr/bin/env python3
"""
Corrected Model Optimization
Uses proper Stratified CV by Selection Count for realistic performance estimation
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data with proper stratification"""
    print("[DATA] Loading data...")
    
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
    
    # Aggregate data
    grouping_cols = [
        'employee_shop', 'employee_branch', 'employee_gender',
        'product_main_category', 'product_sub_category', 'product_brand',
        'product_color', 'product_durability', 'product_target_gender',
        'product_utility_type', 'product_type'
    ]
    
    agg_data = data.groupby(grouping_cols).size().reset_index(name='selection_count')
    print(f"[AGGREGATE] {len(data)} events -> {len(agg_data)} combinations")
    
    # Create stratification categories
    agg_data['selection_strata'] = pd.cut(
        agg_data['selection_count'], 
        bins=[0, 1, 2, 5, 10, np.inf], 
        labels=[0, 1, 2, 3, 4]  # Use integers for stratification
    )
    
    # Prepare features
    X = agg_data[grouping_cols].copy()
    y = agg_data['selection_count']
    y_log = np.log1p(y)
    strata = agg_data['selection_strata']
    
    # Label encode
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    print(f"[FEATURES] {X.shape}, Target distribution:")
    print(f"   Mean: {y.mean():.2f}, Std: {y.std():.2f}")
    print(f"   Strata distribution: {strata.value_counts().sort_index().to_dict()}")
    
    return X, y, y_log, strata

def test_model_corrected_cv(name, model, X, y, strata):
    """Test model with corrected stratified CV"""
    print(f"\n[TEST] {name}")
    
    # Traditional validation split for comparison
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    r2_val = r2_score(y_val, y_pred)
    mae_val = mean_absolute_error(y_val, y_pred)
    
    # Corrected stratified cross-validation
    cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv_stratified.split(X, strata), scoring='r2')
    r2_cv = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Regular CV for comparison
    cv_scores_regular = cross_val_score(model, X, y, cv=5, scoring='r2')
    r2_cv_regular = cv_scores_regular.mean()
    
    # Analysis
    overfitting = r2_val - r2_cv
    
    print(f"  Validation R2: {r2_val:.4f}")
    print(f"  Stratified CV R2: {r2_cv:.4f} +/- {cv_std:.4f}")
    print(f"  Regular CV R2: {r2_cv_regular:.4f} (for comparison)")
    print(f"  Overfitting (Val-StratCV): {overfitting:.4f}")
    print(f"  MAE: {mae_val:.4f}")
    
    if abs(overfitting) < 0.05:
        print("  [EXCELLENT] Very low overfitting")
    elif abs(overfitting) < 0.1:
        print("  [GOOD] Low overfitting")
    elif abs(overfitting) < 0.15:
        print("  [MODERATE] Moderate overfitting")
    else:
        print("  [WARNING] High overfitting")
    
    return r2_val, r2_cv, overfitting

def main():
    """Main corrected optimization"""
    print("="*80)
    print("CORRECTED MODEL OPTIMIZATION - PROPER STRATIFIED CV")
    print("="*80)
    
    # Load and prepare data
    X, y, y_log, strata = load_and_prepare_data()
    
    results = []
    
    # Test 1: XGB with original target
    print("\n" + "="*50)
    print("TESTING WITH CORRECTED STRATIFIED CV")
    print("="*50)
    
    xgb_model = XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.3, reg_lambda=0.3,
        gamma=0.1, min_child_weight=8, random_state=42, n_jobs=-1
    )
    r2_val, r2_cv, overfitting = test_model_corrected_cv("XGB Original Target", xgb_model, X, y, strata)
    results.append(("XGB Original", r2_val, r2_cv, overfitting))
    
    # Test 2: XGB with log target
    xgb_log = XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.3, reg_lambda=0.3,
        gamma=0.1, min_child_weight=8, random_state=42, n_jobs=-1
    )
    r2_val, r2_cv, overfitting = test_model_corrected_cv("XGB Log Target", xgb_log, X, y_log, strata)
    results.append(("XGB Log", r2_val, r2_cv, overfitting))
    
    # Test 3: More conservative XGB
    xgb_conservative = XGBRegressor(
        n_estimators=800, max_depth=5, learning_rate=0.02,
        subsample=0.95, colsample_bytree=0.95, reg_alpha=0.5, reg_lambda=0.5,
        gamma=0.2, min_child_weight=12, random_state=42, n_jobs=-1
    )
    r2_val, r2_cv, overfitting = test_model_corrected_cv("XGB Conservative", xgb_conservative, X, y, strata)
    results.append(("XGB Conservative", r2_val, r2_cv, overfitting))
    
    # Test 4: Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=500, max_depth=12, min_samples_split=10,
        min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1
    )
    r2_val, r2_cv, overfitting = test_model_corrected_cv("Random Forest", rf_model, X, y, strata)
    results.append(("Random Forest", r2_val, r2_cv, overfitting))
    
    # Test 5: Optimized XGB based on diagnostics
    xgb_optimized = XGBRegressor(
        n_estimators=1200, max_depth=7, learning_rate=0.025,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=0.2,
        gamma=0.05, min_child_weight=6, random_state=42, n_jobs=-1
    )
    r2_val, r2_cv, overfitting = test_model_corrected_cv("XGB Optimized", xgb_optimized, X, y, strata)
    results.append(("XGB Optimized", r2_val, r2_cv, overfitting))
    
    # Summary
    print("\n" + "="*80)
    print("CORRECTED OPTIMIZATION RESULTS")
    print("="*80)
    
    # Sort by stratified CV performance (most reliable)
    results.sort(key=lambda x: x[2], reverse=True)
    
    print("Ranking by Stratified CV R2 (most reliable metric):")
    for i, (name, val_r2, cv_r2, overfit) in enumerate(results):
        status = "[STABLE]" if abs(overfit) < 0.1 else "[MODERATE]" if abs(overfit) < 0.15 else "[OVERFIT]"
        print(f"  {i+1}. {name:15} CV R2: {cv_r2:.4f}, Val R2: {val_r2:.4f}, Gap: {overfit:+.4f} {status}")
    
    best_model = results[0]
    print(f"\nBEST MODEL: {best_model[0]}")
    print(f"  Stratified CV R2: {best_model[2]:.4f}")
    print(f"  Validation R2: {best_model[1]:.4f}")
    print(f"  Overfitting: {best_model[3]:+.4f}")
    
    if best_model[2] >= 0.6:
        print("\n[SUCCESS] TARGET ACHIEVED: CV R2 >= 0.6 - PRODUCTION READY!")
    elif best_model[2] >= 0.4:
        print(f"\n[GOOD] CV R2 = {best_model[2]:.4f} - Strong business value potential")
    elif best_model[2] >= 0.25:
        print(f"\n[MODERATE] CV R2 = {best_model[2]:.4f} - Moderate business value")
    else:
        print(f"\n[POOR] CV R2 = {best_model[2]:.4f} - Limited business value")
    
    print("\nKey insights:")
    print("- Stratified CV by selection count provides realistic performance estimates")
    print("- Gap between validation and stratified CV should be < 0.1 for production")
    print("- Performance around 0.25-0.3 R2 indicates moderate predictive value")
    
    print("="*80)
    return best_model[2]

if __name__ == "__main__":
    best_cv_r2 = main()
    print(f"\nFinal Best Stratified CV R2: {best_cv_r2:.4f}")