"""
Test script for Phase 2 - Data Loading & Inspection
Run this after completing Phase 2
"""

import pandas as pd
import os

def test_dataset_exists():
    """Test if dataset file exists"""
    print("Testing dataset file existence...")
    
    dataset_path = 'data/heart.csv'
    
    if os.path.exists(dataset_path):
        print(f"✓ {dataset_path} exists")
        
        # Check file size
        file_size = os.path.getsize(dataset_path)
        print(f"✓ File size: {file_size / 1024:.2f} KB")
        
        print("\n✅ Dataset file found!")
        return True
    else:
        print(f"❌ {dataset_path} not found")
        print("Please download and place the dataset in data/ folder")
        return False

def test_dataset_loading():
    """Test if dataset can be loaded"""
    print("\nTesting dataset loading...")
    
    try:
        df = pd.read_csv('data/heart.csv')
        print(f"✓ Dataset loaded successfully")
        print(f"✓ Shape: {df.shape}")
        print(f"✓ Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        # Verify minimum expected rows
        if df.shape[0] >= 100:
            print(f"✓ Dataset has sufficient data ({df.shape[0]} rows)")
        else:
            print(f"⚠ Dataset seems small ({df.shape[0]} rows)")
        
        print("\n✅ Dataset loading successful!")
        return True, df
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False, None

def test_required_columns():
    """Test if dataset has all required columns"""
    print("\nTesting required columns...")
    
    try:
        df = pd.read_csv('data/heart.csv')
        
        required_columns = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
            'Oldpeak', 'ST_Slope', 'HeartDisease'
        ]
        
        missing_columns = []
        for col in required_columns:
            if col in df.columns:
                print(f"✓ {col} column exists")
            else:
                print(f"❌ {col} column missing")
                missing_columns.append(col)
        
        if not missing_columns:
            print("\n✅ All required columns present!")
            return True
        else:
            print(f"\n❌ Missing columns: {missing_columns}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking columns: {e}")
        return False

def test_target_variable():
    """Test if target variable is correct"""
    print("\nTesting target variable...")
    
    try:
        df = pd.read_csv('data/heart.csv')
        
        if 'HeartDisease' not in df.columns:
            print("❌ Target column 'HeartDisease' not found")
            return False
        
        unique_values = df['HeartDisease'].unique()
        print(f"✓ Target variable 'HeartDisease' found")
        print(f"✓ Unique values: {sorted(unique_values)}")
        
        # Check if binary
        if set(unique_values).issubset({0, 1}):
            print("✓ Target is binary (0 and 1)")
            
            # Show class distribution
            value_counts = df['HeartDisease'].value_counts()
            print(f"\n  Class distribution:")
            print(f"  - No Disease (0): {value_counts.get(0, 0)} ({value_counts.get(0, 0) / len(df) * 100:.1f}%)")
            print(f"  - Disease (1): {value_counts.get(1, 0)} ({value_counts.get(1, 0) / len(df) * 100:.1f}%)")
            
            print("\n✅ Target variable is valid!")
            return True
        else:
            print("❌ Target variable is not binary")
            return False
            
    except Exception as e:
        print(f"❌ Error checking target: {e}")
        return False

def test_data_types():
    """Test if data types are reasonable"""
    print("\nTesting data types...")
    
    try:
        df = pd.read_csv('data/heart.csv')
        
        # Check for expected numeric columns
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'HeartDisease']
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        
        print("✓ Checking numeric columns:")
        for col in numeric_cols:
            if col in df.columns:
                dtype = df[col].dtype
                if dtype in ['int64', 'float64']:
                    print(f"  ✓ {col}: {dtype}")
                else:
                    print(f"  ⚠ {col}: {dtype} (expected numeric)")
        
        print("\n✓ Checking categorical columns:")
        for col in categorical_cols:
            if col in df.columns:
                dtype = df[col].dtype
                if dtype == 'object':
                    print(f"  ✓ {col}: {dtype}")
                else:
                    print(f"  ⚠ {col}: {dtype} (expected object)")
        
        print("\n✅ Data types check completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error checking data types: {e}")
        return False

def run_all_phase2_tests():
    """Run all Phase 2 tests"""
    print("="*60)
    print("PHASE 2 TESTING - Data Loading & Inspection")
    print("="*60)
    
    test1 = test_dataset_exists()
    test2, df = test_dataset_loading()
    test3 = test_required_columns()
    test4 = test_target_variable()
    test5 = test_data_types()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"1. Dataset File Exists:     {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"2. Dataset Loading:         {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"3. Required Columns:        {'✅ PASS' if test3 else '❌ FAIL'}")
    print(f"4. Target Variable:         {'✅ PASS' if test4 else '❌ FAIL'}")
    print(f"5. Data Types:               {'✅ PASS' if test5 else '❌ FAIL'}")
    
    print("\n" + "="*60)
    if test1 and test2 and test3 and test4 and test5:
        print("✅ PHASE 2 COMPLETE - All tests passed!")
        print("You can proceed to Phase 3")
    else:
        print("❌ PHASE 2 INCOMPLETE - Fix the issues above")
    print("="*60)

if __name__ == "__main__":
    run_all_phase2_tests()

