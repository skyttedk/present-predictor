"""
Script to compare the exact input data being passed to the model in API vs direct calls.
This will help us identify where the data preparation differs.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.predictor import get_predictor
from api.schemas.requests import PredictionRequest, GiftItem, Employee
from data.classifier import GiftClassifier

def create_test_data():
    """Create the same test data used in smoke test."""
    # Same data as smoke_test.py
    gifts = [
        GiftItem(product_id="1", description="Wireless Bluetooth Headphones - Black Sony WH-1000XM4"),
        GiftItem(product_id="2", description="Premium Coffee Mug - Ceramic Blue Handle Variant"),
        GiftItem(product_id="3", description="Leather Wallet - Brown Italian Leather Bifold"),
    ]
    
    employees = [
        Employee(name="Lars Andersen"),
        Employee(name="Anna Nielsen"), 
        Employee(name="Michael Kristensen"),
        Employee(name="Sofia Larsen"),
        Employee(name="Henrik Thomsen"),
    ]
    
    return PredictionRequest(
        cvr="28892055",
        branch_no="621000", 
        gifts=gifts,
        employees=employees
    )

def prepare_data_via_api_flow(request: PredictionRequest):
    """Prepare data the same way the API does."""
    from data.cvr_client import CVRClient
    from data.gender_classifier import enhanced_gender_guesser
    
    # Initialize classifier
    classifier = GiftClassifier()
    
    # Step 1: CVR lookup  
    cvr_client = CVRClient()
    company_info = cvr_client.get_company_info(request.cvr)
    
    # Step 2: Classify gifts
    classified_gifts = []
    for gift in request.gifts:
        classified = classifier.classify_gift_attributes(
            present_name=gift.description,
            present_vendor="",
            model_name="",
            model_no=""
        )
        classified_gifts.append({
            'product_id': gift.product_id,
            'classified': classified
        })
    
    # Step 3: Classify employees
    processed_employees = []
    for employee in request.employees:
        gender = enhanced_gender_guesser(employee.name)
        processed_employees.append({
            'name': employee.name,
            'gender': gender
        })
    
    return {
        'company_info': company_info,
        'classified_gifts': classified_gifts,
        'processed_employees': processed_employees,
        'branch_no': request.branch_no
    }

def prepare_data_via_direct_flow(request: PredictionRequest):
    """Prepare data the same way the smoke test does (direct call)."""
    # This mimics what smoke_test.py does when calling predictor directly
    
    # Convert request to the format expected by predictor.predict()
    gifts_data = []
    for gift in request.gifts:
        gifts_data.append({
            'product_id': gift.product_id,
            'description': gift.description
        })
    
    employees_data = []
    for employee in request.employees:
        employees_data.append({
            'name': employee.name
        })
    
    return {
        'branch_no': request.branch_no,
        'gifts': gifts_data,
        'employees': employees_data
    }

def debug_predictor_input(predictor, api_data, direct_data):
    """Debug what gets passed to the actual model in both flows."""
    print("=== API Flow Data ===")
    print(f"Company info: {api_data.get('company_info', {})}")
    print(f"Number of classified gifts: {len(api_data['classified_gifts'])}")
    for i, gift in enumerate(api_data['classified_gifts'][:2]):  # Show first 2
        print(f"  Gift {i+1}: {gift}")
    print(f"Number of processed employees: {len(api_data['processed_employees'])}")
    for i, emp in enumerate(api_data['processed_employees'][:3]):  # Show first 3
        print(f"  Employee {i+1}: {emp}")
    
    print("\n=== Direct Flow Data ===")
    print(f"Branch: {direct_data['branch_no']}")
    print(f"Number of gifts: {len(direct_data['gifts'])}")
    for i, gift in enumerate(direct_data['gifts'][:2]):
        print(f"  Gift {i+1}: {gift}")
    print(f"Number of employees: {len(direct_data['employees'])}")
    for i, emp in enumerate(direct_data['employees'][:3]):
        print(f"  Employee {i+1}: {emp}")

def main():
    print("🔍 Comparing Input Data: API vs Direct Predictor Calls")
    print("=" * 60)
    
    # Create test data
    request = create_test_data()
    print(f"📝 Test data: {len(request.gifts)} gifts, {len(request.employees)} employees")
    
    # Prepare data via both flows
    print("\n🔄 Step 1: Preparing data via API flow...")
    api_data = prepare_data_via_api_flow(request)
    
    print("🔄 Step 2: Preparing data via direct flow...")
    direct_data = prepare_data_via_direct_flow(request)
    
    # Get predictor instance
    print("\n🔄 Step 3: Loading predictor...")
    predictor = get_predictor()
    
    # Debug the input data
    print("\n📊 Step 4: Comparing input data preparation...")
    debug_predictor_input(predictor, api_data, direct_data)
    
    print("\n✅ Data comparison complete!")
    print("\n💡 Next step: Compare the actual feature vectors generated for the model")

if __name__ == "__main__":
    main()