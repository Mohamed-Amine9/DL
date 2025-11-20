"""
Test script for Phase 1 - Environment Setup
Run this after completing Phase 1
"""

import sys
import subprocess

def test_imports():
    """Test if all required libraries are installed"""
    print("Testing library imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
        
        import numpy as np
        print("✓ numpy imported successfully")
        
        import tensorflow as tf
        print(f"✓ tensorflow imported successfully (version: {tf.__version__})")
        
        import sklearn
        print(f"✓ scikit-learn imported successfully (version: {sklearn.__version__})")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
        
        import seaborn as sns
        print("✓ seaborn imported successfully")
        
        print("\n✅ All library imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False

def test_folders():
    """Test if all required folders exist"""
    print("\nTesting folder structure...")
    
    import os
    required_folders = ['data', 'notebooks', 'models', 'report', 'tests']
    
    all_exist = True
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"✓ {folder}/ exists")
        else:
            print(f"❌ {folder}/ missing")
            all_exist = False
    
    if all_exist:
        print("\n✅ All folders exist!")
    else:
        print("\n❌ Some folders are missing!")
    
    return all_exist

def test_notebook_exists():
    """Test if main notebook exists"""
    print("\nTesting notebook existence...")
    
    import os
    notebook_path = 'notebooks/heart_disease_project.ipynb'
    
    if os.path.exists(notebook_path):
        print(f"✓ {notebook_path} exists")
        print("\n✅ Notebook created successfully!")
        return True
    else:
        print(f"❌ {notebook_path} not found")
        return False

def test_dataset_exists():
    """Test if dataset file exists"""
    print("\nTesting dataset file...")
    
    import os
    dataset_path = 'data/heart.csv'
    
    if os.path.exists(dataset_path):
        file_size = os.path.getsize(dataset_path)
        print(f"✓ {dataset_path} exists")
        print(f"✓ File size: {file_size / 1024:.2f} KB")
        print("\n✅ Dataset file found!")
        return True
    else:
        print(f"❌ {dataset_path} not found")
        print("Please copy heart.csv to data/ folder")
        return False

def test_requirements_txt():
    """Test if requirements.txt exists"""
    print("\nTesting requirements.txt...")
    
    import os
    if os.path.exists('requirements.txt'):
        print("✓ requirements.txt exists")
        print("\n✅ requirements.txt found!")
        return True
    else:
        print("❌ requirements.txt not found")
        return False

def run_all_phase1_tests():
    """Run all Phase 1 tests"""
    print("="*60)
    print("PHASE 1 TESTING - Environment & Setup")
    print("="*60)
    
    test1 = test_imports()
    test2 = test_folders()
    test3 = test_notebook_exists()
    test4 = test_dataset_exists()
    test5 = test_requirements_txt()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"1. Library Imports:        {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"2. Folder Structure:        {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"3. Notebook Exists:         {'✅ PASS' if test3 else '❌ FAIL'}")
    print(f"4. Dataset File:           {'✅ PASS' if test4 else '❌ FAIL'}")
    print(f"5. Requirements.txt:       {'✅ PASS' if test5 else '❌ FAIL'}")
    
    print("\n" + "="*60)
    if test1 and test2 and test3 and test4 and test5:
        print("✅ PHASE 1 COMPLETE - All tests passed!")
        print("You can proceed to Phase 2")
    else:
        print("❌ PHASE 1 INCOMPLETE - Fix the issues above")
    print("="*60)

if __name__ == "__main__":
    run_all_phase1_tests()

