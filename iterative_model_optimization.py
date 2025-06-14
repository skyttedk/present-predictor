#!/usr/bin/env python3
"""
Iterative Model Optimization Script
Systematically improves XGBoost R² performance from 0.0922 to target 0.6+
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class ModelOptimizer:
    def __init__(self):
        self.raw_data = None
        self.aggregated_data = None
        self.X = None
        self.y = None
        self.label_encoders = {}
        self.best_r2 = 0.0
        self.best_config = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the 178K dataset"""
        print("[DATA] Loading historical data...")
        
        # Load historical data with robust encoding
        historical_data_path = "src/data/historical/present.selection.historic.csv"
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                self.raw_data = pd.read_csv(historical_data_path, encoding=encoding, dtype='str')
                print(f"[OK] Loaded with {encoding} encoding: {len(self.raw_data)} records")
                break
            except:
                continue
                
        if self.raw_data is None:
            raise ValueError("Could not load data with any encoding")
            
        # Clean data
        print("[CLEAN] Cleaning data...")
        string_columns = self.raw_data.select_dtypes(include=['object']).columns
        for col in string_columns:
            self.raw_data[col] = self.raw_data[col].astype(str).str.strip('"').str.strip()
        
        self.raw_data = self.raw_data.fillna("NONE")
        
        # Standardize categorical values
        categorical_columns = ['employee_gender', 'product_target_gender', 'product_utility_type', 'product_durability', 'product_type']
        for col in categorical_columns:
            if col in self.raw_data.columns:
                self.raw_data[col] = self.raw_data[col].str.lower()
        
        # Memory optimization
        for col in self.raw_data.columns:
            if self.raw_data[col].dtype == 'object':
                unique_ratio = self.raw_data[col].nunique() / len(self.raw_data)
                if unique_ratio < 0.5:
                    self.raw_data[col] = self.raw_data[col].astype('category')
        
        print(f"[INFO] Data cleaning complete: {len(self.raw_data)} records")
        
    def aggregate_data(self):
        """Aggregate selection events"""
        print("[AGGREGATE] Aggregating selection events...")
        
        grouping_columns = [
            'employee_shop', 'employee_branch', 'employee_gender',
            'product_main_category', 'product_sub_category', 'product_brand',
            'product_color', 'product_durability', 'product_target_gender',
            'product_utility_type', 'product_type'
        ]
        
        # Aggregate by counting selection events
        self.aggregated_data = self.raw_data.groupby(grouping_columns, observed=True).size().reset_index(name='selection_count')
        
        compression_ratio = len(self.raw_data) / len(self.aggregated_data)
        print(f"[INFO] Aggregation: {len(self.raw_data)} events -> {len(self.aggregated_data)} combinations ({compression_ratio:.1f}x)")
        
        return grouping_columns
        
    def prepare_features(self, grouping_columns):
        """Prepare features and target"""
        print("[FEATURES] Preparing features...")
        
        # Separate features and target
        self.X = self.aggregated_data[grouping_columns].copy()
        self.y = self.aggregated_data['selection_count']
        
        # Label encode categorical features
        self.label_encoders = {}
        for column in self.X.columns:
            if self.X[column].dtype == 'object' or self.X[column].dtype.name == 'category':
                le = LabelEncoder()
                self.X[column] = le.fit_transform(self.X[column].astype(str))
                self.label_encoders[column] = le
        
        print(f"[INFO] Features: {self.X.shape}, Target: {self.y.shape}")
        print(f"[INFO] Sample-to-feature ratio: {len(self.X) / len(self.X.columns):.1f}:1")
        
    def test_configuration(self, config_name, xgb_params, description=""):
        """Test a specific XGBoost configuration"""
        print(f"\n[TEST] Testing {config_name}: {description}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = XGBRegressor(**xgb_params)
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X, self.y, cv=3, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Feature importance
        max_importance = model.feature_importances_.max()
        
        print(f"[RESULTS]")
        print(f"   R2: {r2:.4f} (CV: {cv_mean:.4f} +/- {cv_std:.4f})")
        print(f"   MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        print(f"   Max Feature Importance: {max_importance:.4f}")
        print(f"   Training Time: {training_time:.1f}s")
        
        # Track best performance
        if r2 > self.best_r2:
            self.best_r2 = r2
            self.best_config = (config_name, xgb_params.copy())
            print(f"[BEST] NEW BEST R2: {r2:.4f}")
            
        return r2, cv_mean, mae, rmse, max_importance
        
    def run_optimization_iterations(self):
        """Run systematic optimization iterations"""
        print("[OPTIMIZE] Starting iterative optimization...")
        
        # Configuration 1: Current optimized parameters
        config1 = {
            'n_estimators': 800,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'gamma': 0.1,
            'min_child_weight': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        self.test_configuration("Current Optimized", config1, "Current optimized parameters")
        
        # Configuration 2: More aggressive parameters
        config2 = {
            'n_estimators': 1200,
            'max_depth': 9,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'gamma': 0.05,
            'min_child_weight': 3,
            'random_state': 42,
            'n_jobs': -1
        }
        self.test_configuration("Aggressive", config2, "More aggressive for complex patterns")
        
        # Configuration 3: Conservative with high regularization
        config3 = {
            'n_estimators': 1500,
            'max_depth': 6,
            'learning_rate': 0.02,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'gamma': 0.2,
            'min_child_weight': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        self.test_configuration("Conservative", config3, "Conservative with regularization")
        
        # Configuration 4: Boosting-focused
        config4 = {
            'n_estimators': 2000,
            'max_depth': 5,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'gamma': 0.1,
            'min_child_weight': 7,
            'random_state': 42,
            'n_jobs': -1
        }
        self.test_configuration("Boosting-Focused", config4, "Many weak learners approach")
        
        # Configuration 5: Tree-focused
        config5 = {
            'n_estimators': 600,
            'max_depth': 12,
            'learning_rate': 0.08,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'gamma': 0.01,
            'min_child_weight': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        self.test_configuration("Tree-Focused", config5, "Deep trees for complex interactions")
        
    def feature_engineering_iteration(self):
        """Test with enhanced feature engineering"""
        print("\n[FEATURES] Testing Enhanced Feature Engineering...")
        
        # Create interaction features
        X_enhanced = self.X.copy()
        
        # Add interaction features
        X_enhanced['category_gender'] = X_enhanced['product_main_category'] * 1000 + X_enhanced['employee_gender']
        X_enhanced['brand_durability'] = X_enhanced['product_brand'] * 1000 + X_enhanced['product_durability']
        X_enhanced['shop_category'] = X_enhanced['employee_shop'] * 1000 + X_enhanced['product_main_category']
        
        # Add aggregated features based on category popularity
        category_counts = self.aggregated_data.groupby('product_main_category')['selection_count'].agg(['mean', 'std', 'count']).fillna(0)
        category_map = {}
        for idx, row in category_counts.iterrows():
            category_map[idx] = (row['mean'], row['std'], row['count'])
        
        # Map back to features
        le_category = self.label_encoders['product_main_category']
        for i, cat_encoded in enumerate(X_enhanced['product_main_category']):
            if cat_encoded < len(le_category.classes_):
                cat_original = le_category.classes_[cat_encoded]
                if cat_original in category_map:
                    X_enhanced.loc[X_enhanced.index[i], 'category_avg'] = category_map[cat_original][0]
                    X_enhanced.loc[X_enhanced.index[i], 'category_std'] = category_map[cat_original][1]
                    X_enhanced.loc[X_enhanced.index[i], 'category_freq'] = category_map[cat_original][2]
        
        # Fill any remaining NaN values
        X_enhanced = X_enhanced.fillna(0)
        
        print(f"[INFO] Enhanced features: {X_enhanced.shape} (added {X_enhanced.shape[1] - self.X.shape[1]} features)")
        
        # Test with enhanced features
        X_train, X_val, y_train, y_val = train_test_split(
            X_enhanced, self.y, test_size=0.2, random_state=42
        )
        
        # Use best configuration found so far
        best_params = self.best_config[1] if self.best_config else {
            'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.03,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.05, 'reg_lambda': 0.05,
            'random_state': 42, 'n_jobs': -1
        }
        
        model = XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        r2_enhanced = r2_score(y_val, y_pred)
        
        print(f"[RESULTS] Enhanced Features R2: {r2_enhanced:.4f}")
        
        if r2_enhanced > self.best_r2:
            self.best_r2 = r2_enhanced
            self.best_config = ("Enhanced Features", best_params)
            print(f"[BEST] NEW BEST with Feature Engineering: {r2_enhanced:.4f}")
        
        return r2_enhanced
        
    def print_final_summary(self):
        """Print final optimization summary"""
        print("\n" + "="*60)
        print("[SUMMARY] OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"[BEST] BEST R2 ACHIEVED: {self.best_r2:.4f}")
        print(f"[INFO] IMPROVEMENT: {((self.best_r2 / 0.0922) - 1) * 100:.1f}% over initial 0.0922")
        
        if self.best_config:
            print(f"[CONFIG] BEST CONFIGURATION: {self.best_config[0]}")
            
        if self.best_r2 >= 0.6:
            print("[STATUS] TARGET ACHIEVED: R2 >= 0.6 - PRODUCTION READY!")
        elif self.best_r2 >= 0.4:
            print("[STATUS] MODERATE: R2 >= 0.4 - Needs more optimization")
        else:
            print("[STATUS] POOR: R2 < 0.4 - Requires advanced techniques")
            
        print("="*60)

def main():
    """Main optimization workflow"""
    optimizer = ModelOptimizer()
    
    # Load and prepare data
    optimizer.load_and_preprocess_data()
    grouping_columns = optimizer.aggregate_data()
    optimizer.prepare_features(grouping_columns)
    
    # Run optimization iterations
    optimizer.run_optimization_iterations()
    
    # Test feature engineering
    optimizer.feature_engineering_iteration()
    
    # Print summary
    optimizer.print_final_summary()
    
    return optimizer.best_r2

if __name__ == "__main__":
    best_r2 = main()
    print(f"\n[FINAL] Best R2: {best_r2:.4f}")