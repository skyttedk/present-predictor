#!/usr/bin/env python3
"""
Quick test script to debug gender classification
"""

from src.data.gender_classifier import classify_employee_gender, classify_employee_gender_detailed

# Test with some Danish names from your Flask example
test_names = [
    "GUNHILD SØRENSEN",
    "Per Christian Eidevik", 
    "Mette",
    "Lars",
    "Bo",
    "Ea",
    "Anne-Marie",
    "Erik Nielsen",
    "John Doe",
    "Jane Smith"
]

print("=== Gender Classification Debug ===")
for name in test_names:
    result = classify_employee_gender_detailed(name)
    print(f"Name: '{name}' -> Gender: {result.gender.value}, Confidence: {result.confidence}, Method: {result.method}")

print("\n=== Simple Classification ===")
for name in test_names:
    gender = classify_employee_gender(name)
    print(f"Name: '{name}' -> Gender: {gender.value}")