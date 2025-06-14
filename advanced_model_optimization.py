#!/usr/bin/env python3
"""
Advanced Model Optimization Script
Addresses overfitting and tests alternative approaches for R² improvement
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelOptimizer:
    def __init__(self):
        self.raw_data = None
        self.aggregated_data = None
        self.X = None
        self.y = None
        self.label_encoders = {}
        self.scaler = None
        self.best_r2 = 0.0
        self.best_config = None
        self.results_log = []
        
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
        
    def aggregate_data_advanced(self):
        """Advanced aggregation with better target engineering"""
        print("[AGGREGATE] Advanced aggregation...")
        
        grouping_columns = [
            'employee_shop', 'employee_branch', 'employee_gender',
            'product_main_category', 'product_sub_category', 'product_brand',
            'product_color', 'product_durability', 'product_target_gender',
            'product_utility_type', 'product_type'
        ]
        
        # Aggregate by counting selection events
        self.aggregated_data = self.raw_data.groupby(grouping_columns, observed=True).size().reset_index(name='selection_count')
        
        # Create log-transformed target to handle skewness
        self.aggregated_data['log_selection_count'] = np.log1p(self.aggregated_data['selection_count'])
        
        # Create categorical target for stratified sampling
        self.aggregated_data['selection_category'] = pd.cut(
            self.aggregated_data['selection_count'], 
            bins=[0, 1, 2, 5, 10, np.inf], 
            labels=['single', 'few', 'some', 'many', 'frequent']
        )
        
        compression_ratio = len(self.raw_data) / len(self.aggregated_data)
        print(f"[INFO] Aggregation: {len(self.raw_data)} events -> {len(self.aggregated_data)} combinations ({compression_ratio:.1f}x)")
        
        # Analyze target distribution
        print(f"[INFO] Target distribution:")
        print(f"   Mean: {self.aggregated_data['selection_count'].mean():.2f}")
        print(f"   Std: {self.aggregated_data['selection_count'].std():.2f}")
        print(f"   Min: {self.aggregated_data['selection_count'].min()}")
        print(f"   Max: {self.aggregated_data['selection_count'].max()}")
        print(f"   Skewness: {self.aggregated_data['selection_count'].skew():.2f}")
        
        return grouping_columns
        
    def prepare_features_advanced(self, grouping_columns):
        """Advanced feature preparation"""
        print("[FEATURES] Advanced feature preparation...")
        
        # Separate features and target
        self.X = self.aggregated_data[grouping_columns].copy()
        
        # Create multiple target variations
        self.y_original = self.aggregated_data['selection_count']
        self.y_log = self.aggregated_data['log_selection_count']
        self.y_category = self.aggregated_data['selection_category']
        
        # Label encode categorical features
        self.label_encoders = {}
        for column in self.X.columns:
            if self.X[column].dtype == 'object' or self.X[column].dtype.name == 'category':
                le = LabelEncoder()
                self.X[column] = le.fit_transform(self.X[column].astype(str))
                self.label_encoders[column] = le
        
        # Add advanced engineered features
        self.X_enhanced = self.create_advanced_features(self.X)
        
        print(f"[INFO] Original features: {self.X.shape}")
        print(f"[INFO] Enhanced features: {self.X_enhanced.shape}")
        print(f"[INFO] Sample-to-feature ratio: {len(self.X_enhanced) / self.X_enhanced.shape[1]:.1f}:1")
        
    def create_advanced_features(self, X):
        """Create advanced engineered features"""
        X_enhanced = X.copy()
        
        # Interaction features
        X_enhanced['category_gender'] = X['product_main_category'] * 1000 + X['employee_gender']
        X_enhanced['brand_durability'] = X['product_brand'] * 1000 + X['product_durability']
        X_enhanced['shop_category'] = X['employee_shop'] * 1000 + X['product_main_category']
        X_enhanced['target_utility'] = X['product_target_gender'] * 100 + X['product_utility_type']
        
        # Frequency encoding features
        for col in ['product_main_category', 'product_brand', 'employee_shop']:
            freq_map = self.aggregated_data.groupby(col)['selection_count'].agg(['mean', 'std', 'count', 'sum']).fillna(0)
            le = self.label_encoders[col]
            
            for i, encoded_val in enumerate(X[col]):
                if encoded_val < len(le.classes_):
                    original_val = le.classes_[encoded_val]
                    if original_val in freq_map.index:
                        X_enhanced.loc[X_enhanced.index[i], f'{col}_mean'] = freq_map.loc[original_val, 'mean']
                        X_enhanced.loc[X_enhanced.index[i], f'{col}_std'] = freq_map.loc[original_val, 'std']
                        X_enhanced.loc[X_enhanced.index[i], f'{col}_count'] = freq_map.loc[original_val, 'count']
                        X_enhanced.loc[X_enhanced.index[i], f'{col}_sum'] = freq_map.loc[original_val, 'sum']
        
        # Fill NaN values
        X_enhanced = X_enhanced.fillna(0)
        
        return X_enhanced
        
    def test_algorithm(self, name, model, X, y, use_scaling=False):
        """Test a specific algorithm"""
        print(f"\n[TEST] Testing {name}")
        
        # Prepare data
        if use_scaling:
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            X_test = X_scaled
        else:
            X_test = X
        
        # Split with stratification if possible
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_test, y, test_size=0.2, random_state=42,
                stratify=self.y_category
            )
        except:
            X_train, X_val, y_train, y_val = train_test_split(
                X_test, y, test_size=0.2, random_state=42
            )
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        # Cross-validation with stratification
        try:
            cv_scores = cross_val_score(
                model, X_test, y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), 
                scoring='r2'
            )
        except:
            cv_scores = cross_val_score(model, X_test, y, cv=3, scoring='r2')
        
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"[RESULTS]")
        print(f"   R2: {r2:.4f} (CV: {cv_mean:.4f} +/- {cv_std:.4f})")
        print(f"   MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        print(f"   Training Time: {training_time:.1f}s")
        print(f"   Overfitting: {(r2 - cv_mean):.4f}")
        
        # Track results
        result = {
            'name': name,
            'r2': r2,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'overfitting': r2 - cv_mean,
            'mae': mae,
            'rmse': rmse,
            'training_time': training_time
        }
        self.results_log.append(result)
        
        # Track best performance
        if r2 > self.best_r2:
            self.best_r2 = r2
            self.best_config = (name, model)
            print(f"[BEST] NEW BEST R2: {r2:.4f}")
            
        return r2, cv_mean
        
    def run_comprehensive_testing(self):
        """Run comprehensive algorithm testing"""
        print("[OPTIMIZE] Starting comprehensive testing...")
        
        # Test 1: XGBoost with original target
        print("\n" + "="*50)
        print("[PHASE 1] XGBoost Variations with Original Target")
        print("="*50)
        
        # Conservative XGBoost (our current best)
        xgb_conservative = XGBRegressor(
            n_estimators=1500, max_depth=6, learning_rate=0.02,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.5, reg_lambda=0.5,
            gamma=0.2, min_child_weight=10, random_state=42, n_jobs=-1
        )
        self.test_algorithm("XGB Conservative (Original)", xgb_conservative, self.X, self.y_original)
        
        # XGBoost with enhanced features
        self.test_algorithm("XGB Conservative (Enhanced)", xgb_conservative, self.X_enhanced, self.y_original)
        
        # Test 2: Log-transformed target
        print("\n" + "="*50)
        print("[PHASE 2] Log-Transformed Target")
        print("="*50)
        
        self.test_algorithm("XGB Conservative (Log Target)", xgb_conservative, self.X_enhanced, self.y_log)
        
        # Test 3: Alternative algorithms
        print("\n" + "="*50)
        print("[PHASE 3] Alternative Algorithms")
        print("="*50)
        
        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=500, max_depth=15, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        self.test_algorithm("Random Forest", rf, self.X_enhanced, self.y_original)
        
        # Gradient Boosting
        gb = GradientBoostingRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        self.test_algorithm("Gradient Boosting", gb, self.X_enhanced, self.y_original)
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
                lgbm = lgb.LGBMRegressor(
                    n_estimators=1000, max_depth=8, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                    random_state=42, n_jobs=-1, verbose=-1
                )
                self.test_algorithm("LightGBM", lgbm, self.X_enhanced, self.y_original)
            except Exception as e:
                print(f"[SKIP] LightGBM error: {e}")
        else:
            print("[SKIP] LightGBM not installed")
        
        # Test 4: Linear models with scaling
        print("\n" + "="*50)
        print("[PHASE 4] Linear Models with Scaling")
        print("="*50)
        
        # Ridge regression
        ridge = Ridge(alpha=1.0, random_state=42)
        self.test_algorithm("Ridge Regression", ridge, self.X_enhanced, self.y_original, use_scaling=True)
        
        # Elastic Net
        elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        self.test_algorithm("Elastic Net", elastic, self.X_enhanced, self.y_original, use_scaling=True)
        
    def print_comprehensive_summary(self):
        """Print comprehensive optimization summary"""
        print("\n" + "="*80)
        print("[SUMMARY] COMPREHENSIVE OPTIMIZATION SUMMARY")
        print("="*80)
        
        # Sort results by R²
        sorted_results = sorted(self.results_log, key=lambda x: x['r2'], reverse=True)
        
        print(f"[BEST] BEST R2 ACHIEVED: {self.best_r2:.4f}")
        print(f"[INFO] IMPROVEMENT: {((self.best_r2 / 0.0922) - 1) * 100:.1f}% over initial 0.0922")
        
        if self.best_config:
            print(f"[CONFIG] BEST CONFIGURATION: {self.best_config[0]}")
        
        print(f"\n[RANKING] Top 5 Performers:")
        for i, result in enumerate(sorted_results[:5]):
            overfitting_flag = "⚠️ HIGH OVERFITTING" if result['overfitting'] > 0.1 else ""
            print(f"   {i+1}. {result['name']}: R2={result['r2']:.4f}, CV={result['cv_mean']:.4f} {overfitting_flag}")
        
        # Overfitting analysis
        print(f"\n[OVERFITTING] Analysis:")
        high_overfitting = [r for r in sorted_results if r['overfitting'] > 0.1]
        low_overfitting = [r for r in sorted_results if r['overfitting'] <= 0.1]
        
        if low_overfitting:
            best_stable = max(low_overfitting, key=lambda x: x['cv_mean'])
            print(f"   Best stable model: {best_stable['name']} (CV R2: {best_stable['cv_mean']:.4f})")
        
        print(f"   High overfitting models: {len(high_overfitting)}/{len(sorted_results)}")
        
        if self.best_r2 >= 0.6:
            print("[STATUS] TARGET ACHIEVED: R2 >= 0.6 - PRODUCTION READY!")
        elif self.best_r2 >= 0.4:
            print("[STATUS] MODERATE: R2 >= 0.4 - Needs more optimization")
        else:
            print("[STATUS] POOR: R2 < 0.4 - May need data strategy review")
            
        print("="*80)

def main():
    """Main optimization workflow"""
    optimizer = AdvancedModelOptimizer()
    
    # Load and prepare data
    optimizer.load_and_preprocess_data()
    grouping_columns = optimizer.aggregate_data_advanced()
    optimizer.prepare_features_advanced(grouping_columns)
    
    # Run comprehensive testing
    optimizer.run_comprehensive_testing()
    
    # Print summary
    optimizer.print_comprehensive_summary()
    
    return optimizer.best_r2

if __name__ == "__main__":
    best_r2 = main()
    print(f"\n[FINAL] Best R2: {best_r2:.4f}")