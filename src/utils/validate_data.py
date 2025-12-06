import pandas as pd
from typing import Tuple, List

def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset.
    """
    print("Starting data validation...")
    
    failed_expectations = []
    
    print(f"Validating schema and required columns...")
    required_columns = [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]
    
    for col in required_columns:
        if col not in df.columns:
            failed_expectations.append(f"Missing column: {col}")
        elif col == "customerID" and df[col].isnull().any():
            failed_expectations.append(f"Null values in {col}")
    
    # === BUSINESS LOGIC VALIDATION ===
    print("Validating business logic constraints...")
    if not df["gender"].isin(["Male", "Female"]).all():
        failed_expectations.append("Invalid values in gender")
    if not df["Partner"].isin(["Yes", "No"]).all():
        failed_expectations.append("Invalid values in Partner")
    if not df["Dependents"].isin(["Yes", "No"]).all():
        failed_expectations.append("Invalid values in Dependents")
    if not df["PhoneService"].isin(["Yes", "No"]).all():
        failed_expectations.append("Invalid values in PhoneService")
    if not df["Contract"].isin(["Month-to-month", "One year", "Two year"]).all():
        failed_expectations.append("Invalid values in Contract")
    if not df["InternetService"].isin(["DSL", "Fiber optic", "No"]).all():
        failed_expectations.append("Invalid values in InternetService")
    
    # === NUMERIC RANGE VALIDATION ===
    print("Validating numeric ranges and business constraints...")
    if (pd.to_numeric(df["tenure"], errors='coerce') < 0).any():
        failed_expectations.append("Negative values in tenure")
    if (pd.to_numeric(df["MonthlyCharges"], errors='coerce') < 0).any():
        failed_expectations.append("Negative values in MonthlyCharges")
    if (pd.to_numeric(df["TotalCharges"], errors='coerce') < 0).any():
        failed_expectations.append("Negative values in TotalCharges")
    
    # === STATISTICAL VALIDATION ===
    print("Validating statistical properties...")
    if (pd.to_numeric(df["tenure"], errors='coerce') > 120).any():
        failed_expectations.append("tenure values exceed 120")
    if (pd.to_numeric(df["MonthlyCharges"], errors='coerce') > 200).any():
        failed_expectations.append("MonthlyCharges values exceed 200")
    if df["tenure"].isnull().any():
        failed_expectations.append("Null values in tenure")
    if df["MonthlyCharges"].isnull().any():
        failed_expectations.append("Null values in MonthlyCharges")
    
    # === DATA CONSISTENCY CHECKS ===
    print("Validating data consistency...")
    total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors='coerce')
    monthly_charges_numeric = pd.to_numeric(df["MonthlyCharges"], errors='coerce')
    consistency_check = (total_charges_numeric >= monthly_charges_numeric).sum() / len(df)
    if consistency_check < 0.95:
        failed_expectations.append("TotalCharges < MonthlyCharges for more than 5% of records")
    
    print("Running complete validation suite...")
    
    total_checks = 15
    failed_checks = len(failed_expectations)
    passed_checks = total_checks - failed_checks
    
    is_valid = len(failed_expectations) == 0
    
    if is_valid:
        print(f"Data validation PASSED: {passed_checks}/{total_checks}")
    else:
        print(f"Data validation FAILED: {failed_checks}/{total_checks}")
        print("Failed expectations:", failed_expectations)
    
    return is_valid, failed_expectations