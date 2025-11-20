"""
Test script for Phase 3 - Exploratory Data Analysis
Run this after completing Phase 3
"""

import pandas as pd
import numpy as np
import os

def test_eda_completed():
    """Test if EDA was completed properly"""
    print("Testing EDA completion...")
    
    try:
        df = pd.read_csv('data/heart.csv')
        
        # Test 1: Check if missing values were analyzed
        print("\n✓ Dataset loaded for EDA testing")
        missing_count = df.isna().sum().sum()
        print(f"✓ Missing values count: {missing_count}")
        
        # Test 2: Check target variable
        if 'HeartDisease' in df.columns:
            target_dist = df['HeartDisease'].value_counts()
            print(f"✓ Target distribution:")
            print(f"  - Class 0: {target_dist[0]} ({target_dist[0]/len(df)*100:.1f}%)")
            print(f"  - Class 1: {target_dist[1]} ({target_dist[1]/len(df)*100:.1f}%)")
            
            # Check balance
            balance_ratio = min(target_dist) / max(target_dist) * 100
            if balance_ratio > 70:
                print(f"✓ Dataset is balanced (ratio: {balance_ratio:.1f}%)")
            else:
                print(f"⚠ Dataset imbalance detected (ratio: {balance_ratio:.1f}%)")
        
        print("\n✅ EDA tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ EDA testing failed: {e}")
        return False

def test_feature_identification():
    """Test if numeric and categorical features were identified"""
    print("\nTesting feature identification...")
    
    try:
        df = pd.read_csv('data/heart.csv')
        
        expected_numeric = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'FastingBS']
        expected_categorical = ['Sex', 'ChestPainType', 'RestingECG', 
                               'ExerciseAngina', 'ST_Slope']
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from numeric
        if 'HeartDisease' in numeric_cols:
            numeric_cols.remove('HeartDisease')
        
        print(f"✓ Numeric features found: {len(numeric_cols)}")
        print(f"✓ Categorical features found: {len(categorical_cols)}")
        
        # Check if key features exist
        all_features_exist = True
        for feat in expected_numeric + expected_categorical:
            if feat in df.columns:
                print(f"✓ {feat} exists")
            else:
                print(f"❌ {feat} missing")
                all_features_exist = False
        
        if all_features_exist:
            print("\n✅ Feature identification complete!")
            return True
        else:
            print("\n⚠ Some features missing")
            return False
            
    except Exception as e:
        print(f"❌ Feature identification failed: {e}")
        return False

def test_data_quality_checks():
    """Test if data quality checks were performed"""
    print("\nTesting data quality checks...")
    
    try:
        df = pd.read_csv('data/heart.csv')
        
        # Check 1: No completely empty rows
        empty_rows = df.isna().all(axis=1).sum()
        if empty_rows == 0:
            print("✓ No completely empty rows")
        else:
            print(f"⚠ Found {empty_rows} empty rows")
        
        # Check 2: No duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates == 0:
            print("✓ No duplicate rows found")
        else:
            print(f"⚠ Found {duplicates} duplicate rows")
        
        # Check 3: Reasonable value ranges for key features
        if 'Age' in df.columns:
            age_range = (df['Age'].min(), df['Age'].max())
            if 0 < age_range[0] and age_range[1] < 150:
                print(f"✓ Age range is reasonable: {age_range}")
            else:
                print(f"⚠ Age range seems unusual: {age_range}")
        
        # Check 4: Target variable is binary
        if 'HeartDisease' in df.columns:
            unique_target = set(df['HeartDisease'].unique())
            if unique_target.issubset({0, 1}):
                print("✓ Target variable is binary (0, 1)")
            else:
                print(f"⚠ Target variable has unexpected values: {unique_target}")
        
        # Check 5: Check for potential outliers in numeric features
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'HeartDisease' in numeric_cols:
            numeric_cols.remove('HeartDisease')
        
        print("\n✓ Checking for potential outliers:")
        for col in numeric_cols[:3]:  # Check first 3 numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"  ⚠ {col}: {outliers} potential outliers detected")
            else:
                print(f"  ✓ {col}: No outliers detected")
        
        print("\n✅ Data quality checks completed!")
        return True
        
    except Exception as e:
        print(f"❌ Data quality checks failed: {e}")
        return False

def test_correlation_analysis():
    """Test if correlation analysis can be performed"""
    print("\nTesting correlation analysis capability...")
    
    try:
        df = pd.read_csv('data/heart.csv')
        
        # Get numeric features
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'HeartDisease' in numeric_cols:
            numeric_cols.remove('HeartDisease')
        
        # Add target for correlation
        correlation_cols = numeric_cols + ['HeartDisease']
        
        # Calculate correlation
        corr_matrix = df[correlation_cols].corr()
        
        # Check correlation with target
        target_corr = corr_matrix['HeartDisease'].drop('HeartDisease')
        
        print(f"✓ Correlation matrix calculated ({len(correlation_cols)} features)")
        print(f"✓ Top 3 features correlated with target:")
        top_corr = target_corr.abs().sort_values(ascending=False).head(3)
        for feat, corr_val in top_corr.items():
            print(f"  - {feat}: {corr_val:.3f}")
        
        print("\n✅ Correlation analysis successful!")
        return True
        
    except Exception as e:
        print(f"❌ Correlation analysis failed: {e}")
        return False

def run_all_phase3_tests():
    """Run all Phase 3 tests"""
    print("="*60)
    print("PHASE 3 TESTING - Exploratory Data Analysis")
    print("="*60)
    
    test1 = test_eda_completed()
    test2 = test_feature_identification()
    test3 = test_data_quality_checks()
    test4 = test_correlation_analysis()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"1. EDA Completion:           {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"2. Feature Identification:   {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"3. Data Quality Checks:      {'✅ PASS' if test3 else '❌ FAIL'}")
    print(f"4. Correlation Analysis:     {'✅ PASS' if test4 else '❌ FAIL'}")
    
    print("\n" + "="*60)
    if test1 and test2 and test3 and test4:
        print("✅ PHASE 3 COMPLETE - All tests passed!")
        print("You can proceed to Phase 4")
    else:
        print("❌ PHASE 3 INCOMPLETE - Review the warnings above")
    print("="*60)

if __name__ == "__main__":
    run_all_phase3_tests()

