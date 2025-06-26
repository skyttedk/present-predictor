#!/usr/bin/env python3
"""
Enhanced Training Pipeline Validation Script

This script tests and validates the enhanced three-file architecture implementation
for the gift selection prediction system. It verifies:

1. Enhanced training pipeline functionality
2. Business logic validation (reasonable selection rates)
3. Zero data leakage verification
4. Enhanced predictor with optimal API format
5. Complete system integration

Usage:
    python test_enhanced_training.py
"""

import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np

try:
    from ml.catboost_trainer_enhanced import EnhancedCatBoostTrainer
    from ml.predictor_enhanced import EnhancedPredictor
    from api.schemas.optimal_requests import (
        OptimalPredictionRequest, 
        OptimalGiftItem,
        BusinessValidationMetrics
    )
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


class EnhancedTrainingValidator:
    """
    Comprehensive validator for the enhanced training pipeline.
    """
    
    def __init__(self):
        self.results = {}
        self.warnings = []
        self.errors = []
        
    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        
        self.results[test_name] = {
            "success": success,
            "details": details
        }
        
        if not success:
            self.errors.append(f"{test_name}: {details}")
    
    def log_warning(self, message: str):
        """Log warning message."""
        print(f"⚠️  WARNING: {message}")
        self.warnings.append(message)
    
    def log_info(self, message: str):
        """Log info message."""
        print(f"ℹ️  {message}")

    def test_data_files_exist(self) -> bool:
        """Test that all required data files exist."""
        try:
            required_files = [
                "src/data/historical/present.selection.historic.csv",
                "src/data/historical/shop.catalog.csv", 
                "src/data/historical/company.employees.csv"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                self.log_result(
                    "Data Files Existence", 
                    False, 
                    f"Missing files: {missing_files}"
                )
                return False
            
            self.log_result("Data Files Existence", True, "All required data files found")
            return True
            
        except Exception as e:
            self.log_result("Data Files Existence", False, f"Error: {str(e)}")
            return False

    def test_data_structure_validation(self) -> bool:
        """Test data structure and format validation."""
        try:
            # Load and validate selections data
            selections_df = pd.read_csv("src/data/historical/present.selection.historic.csv")
            catalog_df = pd.read_csv("src/data/historical/shop.catalog.csv")
            employees_df = pd.read_csv("src/data/historical/company.employees.csv")
            
            # Validate selections structure
            required_selection_cols = [
                'shop_id', 'company_cvr', 'employee_gender', 'gift_id',
                'product_main_category', 'product_sub_category', 'product_brand',
                'product_color', 'product_durability', 'product_target_gender',
                'product_utility_type', 'product_type'
            ]
            
            missing_cols = [col for col in required_selection_cols if col not in selections_df.columns]
            if missing_cols:
                self.log_result(
                    "Data Structure Validation",
                    False,
                    f"Missing columns in selections: {missing_cols}"
                )
                return False
            
            # Validate catalog structure
            if not all(col in catalog_df.columns for col in ['shop_id', 'gift_id']):
                self.log_result(
                    "Data Structure Validation",
                    False,
                    "Missing required columns in catalog"
                )
                return False
            
            # Validate employees structure
            required_emp_cols = ['company_cvr', 'branch_code', 'male_count', 'female_count']
            missing_emp_cols = [col for col in required_emp_cols if col not in employees_df.columns]
            if missing_emp_cols:
                self.log_result(
                    "Data Structure Validation",
                    False,
                    f"Missing columns in employees: {missing_emp_cols}"
                )
                return False
            
            self.log_result(
                "Data Structure Validation", 
                True, 
                f"Selections: {len(selections_df)} rows, Catalog: {len(catalog_df)} rows, Employees: {len(employees_df)} rows"
            )
            return True
            
        except Exception as e:
            self.log_result("Data Structure Validation", False, f"Error: {str(e)}")
            return False

    def test_enhanced_trainer_initialization(self) -> Tuple[bool, Any]:
        """Test enhanced trainer initialization."""
        try:
            trainer = EnhancedCatBoostTrainer()
            
            self.log_result("Enhanced Trainer Initialization", True, "Trainer created successfully")
            return True, trainer
            
        except Exception as e:
            self.log_result("Enhanced Trainer Initialization", False, f"Error: {str(e)}")
            return False, None

    def test_training_data_creation(self, trainer) -> Tuple[bool, Any]:
        """Test training data creation with exposure calculation."""
        try:
            # Load the three data files
            selections_df = pd.read_csv("src/data/historical/present.selection.historic.csv")
            catalog_df = pd.read_csv("src/data/historical/shop.catalog.csv")
            employees_df = pd.read_csv("src/data/historical/company.employees.csv")
            
            # Create training data
            training_data = trainer.create_training_data_with_exposure(
                selections_df, catalog_df, employees_df
            )
            
            # Validate training data structure
            required_cols = [
                'shop_id', 'company_cvr', 'employee_gender', 'gift_id',
                'selection_count', 'exposure', 'selection_rate'
            ]
            
            missing_cols = [col for col in required_cols if col not in training_data.columns]
            if missing_cols:
                self.log_result(
                    "Training Data Creation",
                    False,
                    f"Missing columns: {missing_cols}"
                )
                return False, None
            
            # Validate selection rates
            rates = training_data['selection_rate']
            invalid_rates = len(rates[(rates < 0) | (rates > 1)])
            if invalid_rates > 0:
                self.log_warning(f"Found {invalid_rates} selection rates outside [0,1] range")
            
            # Validate zero-selection records
            zero_selections = len(training_data[training_data['selection_count'] == 0])
            total_records = len(training_data)
            zero_percentage = (zero_selections / total_records) * 100
            
            self.log_result(
                "Training Data Creation",
                True,
                f"Created {total_records} records ({zero_selections} zero-selections, {zero_percentage:.1f}%)"
            )
            
            return True, training_data
            
        except Exception as e:
            self.log_result("Training Data Creation", False, f"Error: {str(e)}")
            return False, None

    def test_model_training(self, trainer, training_data) -> Tuple[bool, Any]:
        """Test model training with the enhanced pipeline."""
        try:
            self.log_info("Starting model training (this may take a minute)...")
            start_time = time.time()
            
            # Train the model
            model_info = trainer.train_model(training_data)
            
            training_time = time.time() - start_time
            
            # Validate model info
            if not model_info or 'model_path' not in model_info:
                self.log_result("Model Training", False, "No model info returned")
                return False, None
            
            # Check if model file exists
            model_path = Path(model_info['model_path'])
            if not model_path.exists():
                self.log_result("Model Training", False, f"Model file not found: {model_path}")
                return False, None
            
            # Validate performance metrics
            metrics = model_info.get('performance_metrics', {})
            cv_r2 = metrics.get('cv_r2_mean', 0)
            
            if cv_r2 < 0:
                self.log_warning(f"Negative CV R² score: {cv_r2:.4f}")
            
            self.log_result(
                "Model Training",
                True,
                f"Trained in {training_time:.1f}s, CV R²: {cv_r2:.4f}"
            )
            
            return True, model_info
            
        except Exception as e:
            self.log_result("Model Training", False, f"Error: {str(e)}")
            return False, None

    def test_enhanced_predictor(self, model_info) -> Tuple[bool, Any]:
        """Test enhanced predictor initialization."""
        try:
            # Initialize predictor with the trained model
            predictor = EnhancedPredictor(model_path=model_info['model_path'])
            
            self.log_result("Enhanced Predictor Initialization", True, "Predictor loaded successfully")
            return True, predictor
            
        except Exception as e:
            self.log_result("Enhanced Predictor Initialization", False, f"Error: {str(e)}")
            return False, None

    def test_optimal_api_prediction(self, predictor) -> bool:
        """Test prediction with optimal API request format."""
        try:
            # Create test request using optimal format
            test_request = OptimalPredictionRequest(
                cvr="28892055",
                male_count=12,
                female_count=11,
                presents=[
                    OptimalGiftItem(
                        id="gift789",
                        description="Tisvilde Pizzaovn",
                        model_name="Tisvilde Pizzaovn",
                        model_no="",
                        vendor="GaveFabrikken"
                    ),
                    OptimalGiftItem(
                        id="gift234",
                        description="Markberg Toiletry Bag",
                        model_name="Premium Black",
                        model_no="MB-2024",
                        vendor="Markberg"
                    )
                ]
            )
            
            # Make prediction
            response = predictor.predict_optimal(test_request)
            
            # Validate response structure
            if not response or 'predictions' not in response:
                self.log_result("Optimal API Prediction", False, "Invalid response structure")
                return False
            
            predictions = response['predictions']
            if len(predictions) != len(test_request.presents):
                self.log_result(
                    "Optimal API Prediction",
                    False,
                    f"Expected {len(test_request.presents)} predictions, got {len(predictions)}"
                )
                return False
            
            # Validate prediction values
            total_employees = test_request.male_count + test_request.female_count
            total_expected_qty = sum(pred['expected_qty'] for pred in predictions)
            overall_rate = total_expected_qty / total_employees
            
            # Business logic validation
            if overall_rate < 0.1 or overall_rate > 5.0:
                self.log_warning(f"Overall selection rate {overall_rate:.2f} may be outside reasonable range")
            
            self.log_result(
                "Optimal API Prediction",
                True,
                f"Predicted {len(predictions)} gifts, total rate: {overall_rate:.2f}"
            )
            
            return True
            
        except Exception as e:
            self.log_result("Optimal API Prediction", False, f"Error: {str(e)}")
            return False

    def test_business_logic_validation(self, predictor) -> bool:
        """Test business logic validation with various scenarios."""
        try:
            test_scenarios = [
                {"cvr": "28892055", "male_count": 5, "female_count": 5, "name": "Small Company"},
                {"cvr": "12233445", "male_count": 25, "female_count": 30, "name": "Medium Company"},
                {"cvr": "34446505", "male_count": 1, "female_count": 2, "name": "Very Small Company"},
            ]
            
            all_rates_reasonable = True
            
            for scenario in test_scenarios:
                test_request = OptimalPredictionRequest(
                    cvr=scenario["cvr"],
                    male_count=scenario["male_count"],
                    female_count=scenario["female_count"],
                    presents=[
                        OptimalGiftItem(
                            id="gift789",
                            description="Test Gift",
                            model_name="Test Model",
                            model_no="TM-001",
                            vendor="TestVendor"
                        )
                    ]
                )
                
                response = predictor.predict_optimal(test_request)
                predictions = response['predictions']
                
                total_employees = scenario["male_count"] + scenario["female_count"]
                total_expected_qty = sum(pred['expected_qty'] for pred in predictions)
                overall_rate = total_expected_qty / total_employees
                
                # Check if rate is reasonable (0.1 to 3.0 selections per employee)
                rate_reasonable = 0.1 <= overall_rate <= 3.0
                
                if not rate_reasonable:
                    all_rates_reasonable = False
                    self.log_warning(
                        f"{scenario['name']}: Rate {overall_rate:.2f} outside reasonable range"
                    )
                else:
                    self.log_info(
                        f"{scenario['name']}: Rate {overall_rate:.2f} is reasonable"
                    )
            
            self.log_result(
                "Business Logic Validation",
                all_rates_reasonable,
                "All prediction rates within reasonable business ranges" if all_rates_reasonable else "Some rates outside reasonable ranges"
            )
            
            return all_rates_reasonable
            
        except Exception as e:
            self.log_result("Business Logic Validation", False, f"Error: {str(e)}")
            return False

    def test_data_leakage_prevention(self, trainer) -> bool:
        """Test that the training pipeline prevents data leakage."""
        try:
            # Load the data files
            selections_df = pd.read_csv("src/data/historical/present.selection.historic.csv")
            catalog_df = pd.read_csv("src/data/historical/shop.catalog.csv")
            employees_df = pd.read_csv("src/data/historical/company.employees.csv")
            
            # Create training data
            training_data = trainer.create_training_data_with_exposure(
                selections_df, catalog_df, employees_df
            )
            
            # Check for circular dependencies
            leakage_issues = []
            
            # Check 1: Exposure should be external metadata, not derived from selections
            unique_exposures = training_data.groupby(['company_cvr', 'employee_gender'])['exposure'].nunique()
            if (unique_exposures > 1).any():
                leakage_issues.append("Exposure varies for same company-gender combination")
            
            # Check 2: Zero-selection records should exist
            zero_selections = training_data[training_data['selection_count'] == 0]
            if len(zero_selections) == 0:
                leakage_issues.append("No zero-selection records found - model may have selection bias")
            
            # Check 3: Exposure should come from employee metadata, not selection counts
            exposure_from_employees = employees_df.groupby('company_cvr').agg({
                'male_count': 'first',
                'female_count': 'first'
            })
            
            exposure_from_training = training_data.groupby(['company_cvr', 'employee_gender'])['exposure'].first().unstack(fill_value=0)
            
            # Compare male counts
            if 'male' in exposure_from_training.columns:
                male_diff = abs(exposure_from_employees['male_count'] - exposure_from_training['male']).sum()
                if male_diff > 0:
                    leakage_issues.append(f"Male exposure mismatch: {male_diff}")
            
            # Compare female counts  
            if 'female' in exposure_from_training.columns:
                female_diff = abs(exposure_from_employees['female_count'] - exposure_from_training['female']).sum()
                if female_diff > 0:
                    leakage_issues.append(f"Female exposure mismatch: {female_diff}")
            
            if leakage_issues:
                self.log_result(
                    "Data Leakage Prevention",
                    False,
                    f"Potential leakage issues: {leakage_issues}"
                )
                return False
            
            self.log_result(
                "Data Leakage Prevention",
                True,
                "No data leakage detected - exposure correctly sourced from external metadata"
            )
            return True
            
        except Exception as e:
            self.log_result("Data Leakage Prevention", False, f"Error: {str(e)}")
            return False

    def print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "="*60)
        print("🎯 ENHANCED TRAINING PIPELINE VALIDATION SUMMARY")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Tests Run: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Enhanced training pipeline is ready for production use")
            print("✅ Business logic validation successful")
            print("✅ Zero data leakage confirmed")
            print("✅ Optimal API request format working correctly")
        else:
            print(f"\n⚠️  {failed_tests} tests failed - review issues before proceeding")
        
        print("="*60)


def main():
    """Main test execution function."""
    print("🚀 Starting Enhanced Training Pipeline Validation")
    print("="*60)
    
    validator = EnhancedTrainingValidator()
    
    # Test sequence
    tests_passed = 0
    total_tests = 8
    
    # Test 1: Data files exist
    if validator.test_data_files_exist():
        tests_passed += 1
    else:
        print("❌ Critical error: Data files missing. Cannot continue.")
        validator.print_summary()
        return False
    
    # Test 2: Data structure validation
    if validator.test_data_structure_validation():
        tests_passed += 1
    else:
        print("❌ Critical error: Data structure invalid. Cannot continue.")
        validator.print_summary()
        return False
    
    # Test 3: Enhanced trainer initialization
    trainer_success, trainer = validator.test_enhanced_trainer_initialization()
    if trainer_success:
        tests_passed += 1
    else:
        validator.print_summary()
        return False
    
    # Test 4: Training data creation
    training_data_success, training_data = validator.test_training_data_creation(trainer)
    if training_data_success:
        tests_passed += 1
    else:
        validator.print_summary()
        return False
    
    # Test 5: Model training
    model_success, model_info = validator.test_model_training(trainer, training_data)
    if model_success:
        tests_passed += 1
    else:
        validator.print_summary()
        return False
    
    # Test 6: Enhanced predictor
    predictor_success, predictor = validator.test_enhanced_predictor(model_info)
    if predictor_success:
        tests_passed += 1
    else:
        validator.print_summary()
        return False
    
    # Test 7: Optimal API prediction
    if validator.test_optimal_api_prediction(predictor):
        tests_passed += 1
    
    # Test 8: Business logic validation
    if validator.test_business_logic_validation(predictor):
        tests_passed += 1
    
    # Additional test: Data leakage prevention
    validator.test_data_leakage_prevention(trainer)
    
    # Print summary
    validator.print_summary()
    
    # Return success status
    return tests_passed == total_tests


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)