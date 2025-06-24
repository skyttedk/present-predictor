#!/usr/bin/env python3
"""
Proof of Concept: ML Prediction Logic
Tests the core prediction flow with the CatBoost model
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.feature_extraction import FeatureHasher
import os
import json
from typing import List, Dict, Optional
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
MODEL_PATH = "models/catboost_rmse_model/catboost_rmse_model.cbm"
HISTORICAL_DATA_PATH = "src/data/historical/present.selection.historic.csv"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockShopFeatureResolver:
    """Simplified shop feature resolver for POC"""
    
    def __init__(self, historical_data_path: Optional[str] = None):
        self.historical_features = {}
        self.branch_mapping = {}
        
        if historical_data_path and os.path.exists(historical_data_path):
            self._load_historical_features(historical_data_path)
        else:
            logger.warning("Historical data not found, using mock features")
            
        # Default features for testing
        self.default_features = {
            'shop_main_category_diversity_selected': 5,
            'shop_brand_diversity_selected': 8,
            'shop_utility_type_diversity_selected': 3,
            'shop_sub_category_diversity_selected': 6,
            'shop_most_frequent_main_category_selected': 'Home & Kitchen',
            'shop_most_frequent_brand_selected': 'Fiskars',
            'unique_product_combinations_in_shop': 45
        }
    
    def _load_historical_features(self, data_path: str):
        """Load and compute shop features from historical data"""
        try:
            logger.info(f"Loading historical data from {data_path}")
            df = pd.read_csv(data_path, dtype=str)
            
            # Clean data
            for col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.strip('"')
            
            df = df.fillna("NONE")
            
            # Compute shop features
            shop_stats = df.groupby('employee_shop').agg({
                'product_main_category': 'nunique',
                'product_sub_category': 'nunique', 
                'product_brand': 'nunique',
                'product_utility_type': 'nunique'
            }).reset_index()
            
            # Get most frequent categories and brands per shop
            shop_main_cats = df.groupby(['employee_shop', 'product_main_category']).size().reset_index(name='count')
            shop_main_cats = shop_main_cats.loc[shop_main_cats.groupby('employee_shop')['count'].idxmax()]
            
            shop_brands = df.groupby(['employee_shop', 'product_brand']).size().reset_index(name='count')
            shop_brands = shop_brands.loc[shop_brands.groupby('employee_shop')['count'].idxmax()]
            
            # Store features by shop
            for _, row in shop_stats.iterrows():
                shop_id = row['employee_shop']
                
                # Get most frequent items for this shop
                most_freq_cat = shop_main_cats[shop_main_cats['employee_shop'] == shop_id]['product_main_category'].iloc[0] if len(shop_main_cats[shop_main_cats['employee_shop'] == shop_id]) > 0 else 'NONE'
                most_freq_brand = shop_brands[shop_brands['employee_shop'] == shop_id]['product_brand'].iloc[0] if len(shop_brands[shop_brands['employee_shop'] == shop_id]) > 0 else 'NONE'
                
                self.historical_features[shop_id] = {
                    'shop_main_category_diversity_selected': row['product_main_category'],
                    'shop_brand_diversity_selected': row['product_brand'],
                    'shop_utility_type_diversity_selected': row['product_utility_type'],
                    'shop_sub_category_diversity_selected': row['product_sub_category'],
                    'shop_most_frequent_main_category_selected': most_freq_cat,
                    'shop_most_frequent_brand_selected': most_freq_brand,
                    'unique_product_combinations_in_shop': len(df[df['employee_shop'] == shop_id])
                }
            
            # Build branch to shop mapping
            branch_shop_map = df.groupby('employee_branch')['employee_shop'].unique().to_dict()
            for branch, shops in branch_shop_map.items():
                self.branch_mapping[branch] = list(shops)
                
            logger.info(f"Loaded features for {len(self.historical_features)} shops")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def get_shop_features(self, shop_id: str, branch: str) -> Dict:
        """Get shop features with fallback to similar shops"""
        logger.info(f"Getting features for shop {shop_id}, branch {branch}")
        
        # Try direct lookup
        if shop_id in self.historical_features:
            logger.info(f"Found direct features for shop {shop_id}")
            return self.historical_features[shop_id]
        
        # Fallback to similar shops by branch
        if branch in self.branch_mapping:
            similar_shops = self.branch_mapping[branch]
            logger.info(f"Using features from {len(similar_shops)} similar shops in branch {branch}")
            
            # Average features from similar shops
            numeric_features = ['shop_main_category_diversity_selected', 'shop_brand_diversity_selected', 
                              'shop_utility_type_diversity_selected', 'shop_sub_category_diversity_selected',
                              'unique_product_combinations_in_shop']
            
            avg_features = {}
            for feature in numeric_features:
                values = [self.historical_features[s][feature] for s in similar_shops if s in self.historical_features]
                avg_features[feature] = int(np.mean(values)) if values else self.default_features[feature]
            
            # Use most common categorical features
            cat_features = ['shop_most_frequent_main_category_selected', 'shop_most_frequent_brand_selected']
            for feature in cat_features:
                values = [self.historical_features[s][feature] for s in similar_shops if s in self.historical_features]
                if values:
                    avg_features[feature] = max(set(values), key=values.count)
                else:
                    avg_features[feature] = self.default_features[feature]
            
            return avg_features
        
        # Ultimate fallback
        logger.info(f"Using default features for shop {shop_id}")
        return self.default_features.copy()

class SimplifiedPredictor:
    """Simplified predictor for POC testing"""
    
    def __init__(self, model_path: str, historical_data_path: Optional[str] = None):
        logger.info(f"Loading model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)
        self.shop_resolver = MockShopFeatureResolver(historical_data_path)
        self.hasher = FeatureHasher(n_features=10, input_type='string')
        
        # Expected feature columns (from training pipeline)
        self.expected_columns = [
            'employee_shop', 'employee_branch', 'employee_gender',
            'product_main_category', 'product_sub_category', 'product_brand',
            'product_color', 'product_durability', 'product_target_gender',
            'product_utility_type', 'product_type',
            'shop_main_category_diversity_selected', 'shop_brand_diversity_selected',
            'shop_utility_type_diversity_selected', 'shop_sub_category_diversity_selected',
            'shop_most_frequent_main_category_selected', 'shop_most_frequent_brand_selected',
            'unique_product_combinations_in_shop',
            'is_shop_most_frequent_main_category', 'is_shop_most_frequent_brand'
        ] + [f'interaction_hash_{i}' for i in range(10)]
        
        # Categorical feature names (must match training)
        self.categorical_features = [
            'employee_shop', 'employee_branch', 'employee_gender',
            'product_main_category', 'product_sub_category', 'product_brand',
            'product_color', 'product_durability', 'product_target_gender',
            'product_utility_type', 'product_type',
            'shop_most_frequent_main_category_selected', 'shop_most_frequent_brand_selected'
        ]
        
        logger.info(f"Model loaded successfully")
    
    def create_feature_vector(self, present: Dict, employee_gender: str, 
                            shop_id: str, branch: str, shop_features: Dict) -> Dict:
        """Create a single feature vector"""
        
        # Base features
        features = {
            'employee_shop': shop_id,
            'employee_branch': branch,
            'employee_gender': employee_gender,
            'product_main_category': present['item_main_category'],
            'product_sub_category': present['item_sub_category'],
            'product_brand': present['brand'],
            'product_color': present['color'],
            'product_durability': present['durability'],
            'product_target_gender': present['target_demographic'],
            'product_utility_type': present['utility_type'],
            'product_type': present['usage_type']
        }
        
        # Add shop features
        features.update(shop_features)
        
        # Add binary features
        features['is_shop_most_frequent_main_category'] = int(
            present['item_main_category'] == shop_features.get('shop_most_frequent_main_category_selected')
        )
        features['is_shop_most_frequent_brand'] = int(
            present['brand'] == shop_features.get('shop_most_frequent_brand_selected')
        )
        
        return features
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add hashed interaction features"""
        # Create interaction string
        interactions = df.apply(
            lambda x: f"{x['employee_branch']}_{x['product_main_category']}", 
            axis=1
        )
        
        # Hash interactions
        hashed_features = self.hasher.transform(
            interactions.apply(lambda x: [x])
        ).toarray()
        
        # Add as columns
        for i in range(hashed_features.shape[1]):
            df[f'interaction_hash_{i}'] = hashed_features[:, i]
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure features match model expectations"""
        from catboost import Pool
        
        # Convert categorical columns to string
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Ensure numeric columns
        numeric_cols = [
            'shop_main_category_diversity_selected', 'shop_brand_diversity_selected',
            'shop_utility_type_diversity_selected', 'shop_sub_category_diversity_selected',
            'unique_product_combinations_in_shop',
            'is_shop_most_frequent_main_category', 'is_shop_most_frequent_brand'
        ] + [f'interaction_hash_{i}' for i in range(10)]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Add missing columns with defaults
        for col in self.expected_columns:
            if col not in df.columns:
                if col.startswith('interaction_hash_'):
                    df[col] = 0.0
                elif col in self.categorical_features:
                    df[col] = "NONE"
                else:
                    df[col] = 0
        
        # Reorder columns to match training
        df = df[self.expected_columns]
        
        return df
    
    def predict_for_present(self, present: Dict, employees: List[Dict], 
                          shop_id: str, branch: str) -> Dict:
        """Predict quantity for a single present"""
        
        logger.info(f"Predicting for present ID: {present['id']}")
        logger.info(f"  Category: {present['item_main_category']}")
        logger.info(f"  Brand: {present['brand']}")
        
        # Get shop features
        shop_features = self.shop_resolver.get_shop_features(shop_id, branch)
        
        # Count employees by gender
        gender_counts = {'male': 0, 'female': 0, 'unisex': 0}
        for emp in employees:
            gender_counts[emp['gender']] += 1
        
        total_employees = len(employees)
        logger.info(f"  Employees: {total_employees} total ({gender_counts})")
        
        # Create feature vectors for each gender group
        rows = []
        for gender, count in gender_counts.items():
            if count > 0:
                features = self.create_feature_vector(
                    present, gender, shop_id, branch, shop_features
                )
                features['employee_ratio'] = count / total_employees
                rows.append(features)
        
        if not rows:
            logger.warning("No valid employee data found")
            return {
                'product_id': str(present['id']),
                'expected_qty': 0,
                'confidence_score': 0.0,
                'details': {'error': 'No valid employees'}
            }
        
        # Create DataFrame
        feature_df = pd.DataFrame(rows)
        
        # Add interaction features
        feature_df = self.add_interaction_features(feature_df)
        
        # Prepare features for model
        employee_ratios = feature_df['employee_ratio'].values
        feature_df = feature_df.drop(columns=['employee_ratio'])
        feature_df = self.prepare_features(feature_df)
        
        logger.info(f"  Feature shape: {feature_df.shape}")
        
        try:
            # Create CatBoost Pool with proper categorical features
            from catboost import Pool
            
            # Get categorical feature indices
            cat_feature_indices = [
                feature_df.columns.get_loc(col) for col in self.categorical_features
                if col in feature_df.columns
            ]
            
            logger.info(f"  Categorical feature indices: {cat_feature_indices}")
            
            # Create Pool for prediction
            test_pool = Pool(
                data=feature_df,
                cat_features=cat_feature_indices
            )
            
            # Make predictions
            predictions = self.model.predict(test_pool)
            predictions = np.maximum(0, predictions)  # Ensure non-negative
            
            # Aggregate predictions
            # Weight by employee ratio and sum
            weighted_predictions = predictions * employee_ratios
            total_prediction = np.sum(weighted_predictions) * total_employees
            
            # Calculate confidence (simplified)
            if len(predictions) > 1:
                prediction_std = np.std(predictions)
                prediction_mean = np.mean(predictions)
                confidence = min(0.95, 0.7 + (0.25 * (1 - prediction_std / (prediction_mean + 1e-6))))
            else:
                confidence = 0.8  # Default confidence for single prediction
            
            result = {
                'product_id': str(present['id']),
                'expected_qty': int(round(max(0, total_prediction))),
                'confidence_score': round(confidence, 2),
                'details': {
                    'raw_predictions': predictions.tolist(),
                    'gender_distribution': gender_counts,
                    'weighted_sum': float(np.sum(weighted_predictions)),
                    'shop_features_used': shop_features
                }
            }
            
            logger.info(f"  Prediction: {result['expected_qty']} units (confidence: {result['confidence_score']})")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'product_id': str(present['id']),
                'expected_qty': 0,
                'confidence_score': 0.0,
                'details': {'error': str(e)}
            }

def test_prediction():
    """Test the prediction logic with sample data"""
    
    print("=== ML Prediction POC Test ===\n")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please ensure the CatBoost model is trained and saved.")
        return
    
    try:
        # Initialize predictor
        predictor = SimplifiedPredictor(MODEL_PATH, HISTORICAL_DATA_PATH)
        
        # Sample input data (matching the example from the task)
        branch = "621000"
        shop_id = "2960"  # Extract from branch or use mapping
        
        presents = [
            {
                "id": 147748,
                "item_main_category": "Travel",
                "item_sub_category": "Luggage",
                "color": "hvid",
                "brand": "Cavalluzzi",
                "vendor": "NONE",
                "target_demographic": "unisex",
                "utility_type": "practical",
                "usage_type": "individual",
                "durability": "durable"
            },
            {
                "id": 147757,
                "item_main_category": "Travel",
                "item_sub_category": "backpack",
                "color": "NONE",
                "brand": "Bjørn Borg",
                "vendor": "Bjørn Borg",
                "target_demographic": "unisex",
                "utility_type": "practical",
                "usage_type": "individual",
                "durability": "durable"
            },
            {
                "id": 147758,
                "item_main_category": "Home & Kitchen",
                "item_sub_category": "BBQ/Grill & Cutlery",
                "color": "NONE",
                "brand": "Laguiole",
                "vendor": "Fun Nordic",
                "target_demographic": "unisex",
                "utility_type": "practical",
                "usage_type": "individual",
                "durability": "durable"
            }
        ]
        
        employees = [
            {"gender": "female"},
            {"gender": "female"},
            {"gender": "female"},
            {"gender": "male"},
            {"gender": "male"}
        ]
        
        print(f"Testing with {len(presents)} presents and {len(employees)} employees")
        print(f"Shop ID: {shop_id}, Branch: {branch}\n")
        
        # Make predictions
        results = []
        total_prediction = 0
        
        for present in presents:
            result = predictor.predict_for_present(present, employees, shop_id, branch)
            results.append(result)
            total_prediction += result['expected_qty']
            print()
        
        # Summary
        print("=== PREDICTION SUMMARY ===")
        print(f"Total predicted demand: {total_prediction} units")
        print("\nDetailed results:")
        
        for result in results:
            print(f"Present {result['product_id']}: {result['expected_qty']} units (confidence: {result['confidence_score']})")
        
        # Create final response format
        print("\n=== FINAL API RESPONSE FORMAT ===")
        api_response = [
            {
                "id": result['product_id'],
                "quantity": result['expected_qty']
            }
            for result in results
        ]
        
        print(json.dumps(api_response, indent=2))
        
        return results
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()