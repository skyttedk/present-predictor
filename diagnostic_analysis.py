#!/usr/bin/env python3
"""
Diagnostic Analysis Script
Investigates the overfitting gap between validation R² (0.25+) and CV R² (0.05-0.12)
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and preprocess data"""
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
    return data

def analyze_feature_cardinality(data):
    """Analyze cardinality of each feature"""
    print("\n" + "="*60)
    print("FEATURE CARDINALITY ANALYSIS")
    print("="*60)
    
    grouping_cols = [
        'employee_shop', 'employee_branch', 'employee_gender',
        'product_main_category', 'product_sub_category', 'product_brand',
        'product_color', 'product_durability', 'product_target_gender',
        'product_utility_type', 'product_type'
    ]
    
    print("Feature cardinality (unique values):")
    for col in grouping_cols:
        unique_count = data[col].nunique()
        total_count = len(data)
        ratio = unique_count / total_count
        print(f"  {col:25} {unique_count:6} unique ({ratio:.4f} ratio)")
        
        if ratio > 0.1:  # High cardinality features
            print(f"    [WARNING] HIGH CARDINALITY - potential overfitting source")
    
    return grouping_cols

def test_without_high_cardinality_features(data, grouping_cols):
    """Test model performance without high cardinality features"""
    print("\n" + "="*60)
    print("TESTING WITHOUT HIGH CARDINALITY FEATURES")
    print("="*60)
    
    # Aggregate data
    agg_data = data.groupby(grouping_cols).size().reset_index(name='selection_count')
    
    # Identify high cardinality features (>10% unique ratio)
    high_card_features = []
    for col in grouping_cols:
        unique_ratio = data[col].nunique() / len(data)
        if unique_ratio > 0.1:
            high_card_features.append(col)
    
    print(f"High cardinality features to remove: {high_card_features}")
    
    # Create reduced feature set
    reduced_cols = [col for col in grouping_cols if col not in high_card_features]
    print(f"Testing with {len(reduced_cols)} features: {reduced_cols}")
    
    if len(reduced_cols) < 3:
        print("[SKIP] Too few features remaining after removal")
        return None, None
    
    # Re-aggregate with reduced features
    agg_data_reduced = data.groupby(reduced_cols).size().reset_index(name='selection_count')
    
    print(f"Aggregation: {len(data)} events -> {len(agg_data_reduced)} combinations")
    
    # Prepare features
    X = agg_data_reduced[reduced_cols].copy()
    y = agg_data_reduced['selection_count']
    
    # Label encode
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Test model
    return test_model_performance("Reduced Features (No High Cardinality)", X, y)

def test_different_aggregation_methods(data, grouping_cols):
    """Test different ways of aggregating the data"""
    print("\n" + "="*60)
    print("TESTING DIFFERENT AGGREGATION METHODS")
    print("="*60)
    
    results = []
    
    # Method 1: Standard counting (current approach)
    print("\n[METHOD 1] Standard Selection Counting")
    agg_data = data.groupby(grouping_cols).size().reset_index(name='selection_count')
    X, y = prepare_features_simple(agg_data, grouping_cols)
    r2_val, r2_cv = test_model_performance("Standard Counting", X, y)
    results.append(("Standard Counting", r2_val, r2_cv))
    
    # Method 2: Binary occurrence (was this combination selected at least once?)
    print("\n[METHOD 2] Binary Occurrence (0/1)")
    agg_data_binary = data.groupby(grouping_cols).size().reset_index(name='selection_count')
    agg_data_binary['occurred'] = 1  # All combinations occurred at least once
    X, y = prepare_features_simple(agg_data_binary, grouping_cols, target_col='occurred')
    r2_val, r2_cv = test_model_performance("Binary Occurrence", X, y)
    results.append(("Binary Occurrence", r2_val, r2_cv))
    
    # Method 3: Log-transformed counts
    print("\n[METHOD 3] Log-Transformed Counts")
    agg_data_log = data.groupby(grouping_cols).size().reset_index(name='selection_count')
    agg_data_log['log_count'] = np.log1p(agg_data_log['selection_count'])
    X, y = prepare_features_simple(agg_data_log, grouping_cols, target_col='log_count')
    r2_val, r2_cv = test_model_performance("Log Counts", X, y)
    results.append(("Log Counts", r2_val, r2_cv))
    
    # Method 4: Product-only features (no employee info)
    print("\n[METHOD 4] Product Features Only (No Employee Info)")
    product_cols = [col for col in grouping_cols if col.startswith('product_')]
    if len(product_cols) >= 3:
        agg_data_product = data.groupby(product_cols).size().reset_index(name='selection_count')
        X, y = prepare_features_simple(agg_data_product, product_cols)
        r2_val, r2_cv = test_model_performance("Product Only", X, y)
        results.append(("Product Only", r2_val, r2_cv))
    
    return results

def test_cv_methodologies(data, grouping_cols):
    """Test different cross-validation strategies"""
    print("\n" + "="*60)
    print("TESTING CROSS-VALIDATION METHODOLOGIES")
    print("="*60)
    
    # Standard aggregation
    agg_data = data.groupby(grouping_cols).size().reset_index(name='selection_count')
    X, y = prepare_features_simple(agg_data, grouping_cols)
    
    # Model setup
    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.3, reg_lambda=0.3,
        random_state=42, n_jobs=-1
    )
    
    print(f"Testing with {len(X)} samples, {X.shape[1]} features")
    
    # Method 1: Standard K-Fold
    print("\n[CV METHOD 1] Standard 3-Fold")
    cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
    print(f"  CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    
    # Method 2: More folds
    print("\n[CV METHOD 2] 5-Fold")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"  CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    
    # Method 3: Stratified by selection count
    print("\n[CV METHOD 3] Stratified by Selection Count")
    try:
        # Create strata based on selection count
        y_strata = pd.cut(y, bins=[0, 1, 2, 5, np.inf], labels=['single', 'double', 'few', 'many'])
        from sklearn.model_selection import StratifiedKFold
        cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='r2')
        print(f"  CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    except Exception as e:
        print(f"  [ERROR] Stratified CV failed: {e}")
    
    # Method 4: Group by shop (if shop is causing leakage)
    print("\n[CV METHOD 4] Group K-Fold by Shop")
    try:
        # Use shop as groups for GroupKFold
        shop_groups = agg_data['employee_shop']
        cv_scores = cross_val_score(model, X, y, cv=GroupKFold(n_splits=3), groups=shop_groups, scoring='r2')
        print(f"  CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"  [INFO] GroupKFold ensures no shop appears in both train and test")
    except Exception as e:
        print(f"  [ERROR] GroupKFold failed: {e}")

def analyze_data_leakage(data, grouping_cols):
    """Check for potential data leakage issues"""
    print("\n" + "="*60)
    print("DATA LEAKAGE ANALYSIS")
    print("="*60)
    
    # Aggregate data
    agg_data = data.groupby(grouping_cols).size().reset_index(name='selection_count')
    
    # Check if train/test split preserves certain relationships
    X, y = prepare_features_simple(agg_data, grouping_cols)
    
    # Random split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Check feature overlap between train and test
    print("\nFeature value overlap analysis:")
    for i, col in enumerate(grouping_cols):
        train_values = set(X_train.iloc[:, i])
        test_values = set(X_test.iloc[:, i])
        overlap = len(train_values.intersection(test_values))
        train_unique = len(train_values - test_values)
        test_unique = len(test_values - train_values)
        
        print(f"  {col:25} Overlap: {overlap:4}, Train-only: {train_unique:4}, Test-only: {test_unique:4}")
        
        if test_unique > overlap:
            print(f"    [WARNING] Many test values not seen in training")

def prepare_features_simple(agg_data, grouping_cols, target_col='selection_count'):
    """Simple feature preparation"""
    X = agg_data[grouping_cols].copy()
    y = agg_data[target_col]
    
    # Label encode
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y

def test_model_performance(name, X, y):
    """Test model performance with validation and CV"""
    print(f"\n[TEST] {name}")
    
    # Model setup
    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.3, reg_lambda=0.3,
        random_state=42, n_jobs=-1
    )
    
    # Split and train
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Validation performance
    y_pred = model.predict(X_val)
    r2_val = r2_score(y_val, y_pred)
    
    # Cross-validation performance
    cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
    r2_cv = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Overfitting analysis
    overfitting = r2_val - r2_cv
    
    print(f"  Validation R2: {r2_val:.4f}")
    print(f"  CV R2: {r2_cv:.4f} +/- {cv_std:.4f}")
    print(f"  Overfitting gap: {overfitting:.4f}")
    
    if overfitting > 0.15:
        print("  [WARNING] HIGH OVERFITTING")
    elif overfitting < 0.05:
        print("  [GOOD] LOW OVERFITTING")
    else:
        print("  [MODERATE] MODERATE OVERFITTING")
    
    return r2_val, r2_cv

def main():
    """Main diagnostic workflow"""
    print("="*80)
    print("DIAGNOSTIC ANALYSIS: OVERFITTING INVESTIGATION")
    print("="*80)
    
    # Load data
    data = load_data()
    
    # Analyze feature cardinality
    grouping_cols = analyze_feature_cardinality(data)
    
    # Test without high cardinality features
    test_without_high_cardinality_features(data, grouping_cols)
    
    # Test different aggregation methods
    aggregation_results = test_different_aggregation_methods(data, grouping_cols)
    
    # Test CV methodologies
    test_cv_methodologies(data, grouping_cols)
    
    # Analyze data leakage
    analyze_data_leakage(data, grouping_cols)
    
    # Final summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    print("\nAggregation method comparison:")
    if aggregation_results:
        for name, r2_val, r2_cv in aggregation_results:
            gap = r2_val - r2_cv if r2_val and r2_cv else 0
            print(f"  {name:20} Val R2: {r2_val:.4f}, CV R2: {r2_cv:.4f}, Gap: {gap:.4f}")
    
    print("\nKey findings:")
    print("- Check cardinality analysis for overfitting sources")
    print("- Compare different aggregation methods")
    print("- Review CV methodology results")
    print("- Examine data leakage warnings")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()