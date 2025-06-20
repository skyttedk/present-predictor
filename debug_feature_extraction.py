#!/usr/bin/env python3
"""
Debug script to analyze feature extraction for uniform prediction issue.
"""

import sys
sys.path.append('.')

from src.ml.predictor import get_predictor
from src.data.gender_classifier import classify_employee_gender
from src.database.presents import get_present_by_hash
import hashlib
import pandas as pd

def debug_feature_extraction():
    """Debug the feature extraction process to find uniform prediction issue"""
    
    # Sample request data (similar to your production request)
    sample_presents = [
        {"id": 1, "description": "Urban Copenhagen In-ears beige", "model_name": "", "model_no": "", "vendor": ""},
        {"id": 2, "description": "Urban Copenhagen In-ears grøn", "model_name": "", "model_no": "", "vendor": ""},
        {"id": 3, "description": "Different Product Type", "model_name": "Special Model", "model_no": "ABC123", "vendor": "DifferentVendor"},
    ]
    
    sample_employees = [
        {"name": "GUNHILD SØRENSEN"},
        {"name": "Per Christian Eidevik"},
        {"name": "Erik Nielsen"}
    ]
    
    branch = "28892055"
    
    print("🔍 DEBUGGING FEATURE EXTRACTION")
    print("=" * 50)
    
    # Step 1: Check present classification lookup
    print("\n📋 STEP 1: Present Classification Lookup")
    print("-" * 30)
    
    transformed_presents = []
    for present in sample_presents:
        # Use same hash logic as API
        text_to_hash = f"{present['description']} - {present['model_name']} - {present['model_no']}."
        present_hash = hashlib.md5(text_to_hash.encode('utf-8')).hexdigest().upper()
        
        print(f"\nPresent {present['id']}:")
        print(f"  Description: {present['description']}")
        print(f"  Hash: {present_hash}")
        
        # Lookup in database
        attributes = get_present_by_hash(present_hash)
        
        if attributes:
            print(f"  ✅ Found in database")
            for key, value in attributes.items():
                print(f"    {key}: {value}")
                
            present_dict = {
                'id': present['id'],
                'item_main_category': attributes.get('item_main_category', ''),
                'item_sub_category': attributes.get('item_sub_category', ''),
                'color': attributes.get('color', ''),
                'brand': attributes.get('brand', ''),
                'vendor': attributes.get('vendor', ''),
                'target_demographic': attributes.get('target_demographic', ''),
                'utility_type': attributes.get('utility_type', ''),
                'usage_type': attributes.get('usage_type', ''),
                'durability': attributes.get('durability', '')
            }
        else:
            print(f"  ❌ NOT found in database - will get defaults")
            present_dict = {
                'id': present['id'],
                'item_main_category': 'NONE',
                'item_sub_category': 'NONE', 
                'color': 'NONE',
                'brand': 'NONE',
                'vendor': 'NONE',
                'target_demographic': 'unisex',
                'utility_type': 'practical',
                'usage_type': 'individual',
                'durability': 'durable'
            }
        
        transformed_presents.append(present_dict)
    
    # Step 2: Check employee gender classification
    print("\n👥 STEP 2: Employee Gender Classification")
    print("-" * 30)
    
    transformed_employees = []
    for employee in sample_employees:
        gender = classify_employee_gender(employee['name'])
        print(f"  {employee['name']} → {gender.value}")
        transformed_employees.append({'gender': gender.value})
    
    # Step 3: Get predictor and analyze feature generation
    print("\n🤖 STEP 3: ML Feature Generation")
    print("-" * 30)
    
    try:
        predictor = get_predictor()
        
        # Extract shop ID
        shop_id = branch[:4] if len(branch) >= 4 else branch
        print(f"Shop ID: {shop_id}")
        print(f"Branch: {branch}")
        
        # Calculate employee stats
        employee_stats = predictor._calculate_employee_stats(transformed_employees)
        print(f"Employee stats: {employee_stats}")
        
        # Get shop features
        shop_features = predictor.shop_resolver.get_shop_features(shop_id, branch)
        print(f"Shop features: {shop_features}")
        
        # Create feature vectors for first present to analyze
        print(f"\n📊 STEP 4: Feature Vector Analysis (Present 1)")
        print("-" * 30)
        
        present = transformed_presents[0]
        
        # Create features for each gender
        for gender, ratio in employee_stats.items():
            if ratio > 0:
                features = predictor._create_feature_vector(
                    present, gender, shop_id, branch, shop_features
                )
                print(f"\nFeatures for {gender} (ratio: {ratio:.2f}):")
                for key, value in features.items():
                    print(f"  {key}: {value}")
        
        # Check if ALL presents get identical features (the smoking gun)
        print(f"\n🎯 STEP 5: Feature Comparison Across Products")
        print("-" * 30)
        
        feature_vectors = []
        for i, present in enumerate(transformed_presents):
            features = predictor._create_feature_vector(
                present, 'male', shop_id, branch, shop_features
            )
            feature_vectors.append(features)
            print(f"\nPresent {i+1} key features:")
            print(f"  Main Category: {features['product_main_category']}")
            print(f"  Sub Category: {features['product_sub_category']}")
            print(f"  Brand: {features['product_brand']}")
            print(f"  Color: {features['product_color']}")
            print(f"  Target Gender: {features['product_target_gender']}")
            print(f"  Utility Type: {features['product_utility_type']}")
        
        # Check if features are identical
        df = pd.DataFrame(feature_vectors)
        print(f"\n🔍 Feature Uniqueness Check:")
        print("-" * 30)
        
        product_columns = [col for col in df.columns if col.startswith('product_')]
        for col in product_columns:
            unique_values = df[col].nunique()
            print(f"  {col}: {unique_values} unique value(s) → {df[col].unique()}")
        
        if all(df[col].nunique() == 1 for col in product_columns):
            print(f"\n❌ PROBLEM FOUND: All products have IDENTICAL features!")
            print(f"   This explains why predictions are uniform (10-11 units)")
            print(f"   Products are not properly classified or are missing from database")
        else:
            print(f"\n✅ Products have different features - investigating model behavior")
            
    except Exception as e:
        print(f"❌ Error in feature analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_feature_extraction()