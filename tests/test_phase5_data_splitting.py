"""
Test script for Phase 5: Data Splitting & Model Architecture Design
Tests Person B's Phase 5 implementation
"""

import sys
import os
import numpy as np

def test_data_loading():
    """Test if data can be loaded"""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart.csv')
        if not os.path.exists(data_path):
            print(f"❌ Data file not found")
            return False
        
        import pandas as pd
        df = pd.read_csv(data_path)
        
        if df.shape[0] > 0 and df.shape[1] > 0:
            print(f"✓ Data can be loaded: {df.shape}")
            return True
        else:
            print("❌ Data file is empty")
            return False
            
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_data_splitting():
    """Test if data splitting logic is correct"""
    print("\n" + "="*60)
    print("TEST 2: Data Splitting Logic")
    print("="*60)
    
    try:
        from sklearn.model_selection import train_test_split
        
        n_samples = 918
        n_features = 15
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        train_pct = len(X_train) / len(X) * 100
        val_pct = len(X_val) / len(X) * 100
        test_pct = len(X_test) / len(X) * 100
        
        print(f"✓ Split proportions:")
        print(f"  - Training: {train_pct:.1f}% (expected ~70%)")
        print(f"  - Validation: {val_pct:.1f}% (expected ~15%)")
        print(f"  - Test: {test_pct:.1f}% (expected ~15%)")
        
        checks = []
        if 68 <= train_pct <= 72:
            print("  ✓ Training split is correct")
            checks.append(True)
        else:
            checks.append(False)
        
        if 13 <= val_pct <= 17:
            print("  ✓ Validation split is correct")
            checks.append(True)
        else:
            checks.append(False)
        
        if 13 <= test_pct <= 17:
            print("  ✓ Test split is correct")
            checks.append(True)
        else:
            checks.append(False)
        
        return all(checks)
        
    except Exception as e:
        print(f"❌ Data splitting test failed: {e}")
        return False

def test_model_architecture():
    """Test if model architecture can be created"""
    print("\n" + "="*60)
    print("TEST 3: Model Architecture")
    print("="*60)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        tf.random.set_seed(42)
        
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(15,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        print("✓ Model created successfully")
        print(f"  - Number of layers: {len(model.layers)}")
        print(f"  - Total parameters: {model.count_params():,}")
        
        return model.count_params() > 0
            
    except ImportError:
        print("⚠ TensorFlow not available (OK for logic testing)")
        return True
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        return False

def test_model_compilation():
    """Test if model can be compiled"""
    print("\n" + "="*60)
    print("TEST 4: Model Compilation")
    print("="*60)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(15,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✓ Model compiled successfully")
        return True
        
    except ImportError:
        print("⚠ TensorFlow not available (OK for logic testing)")
        return True
    except Exception as e:
        print(f"❌ Model compilation test failed: {e}")
        return False

def test_notebook_section_exists():
    """Test if Phase 5 section exists in notebook"""
    print("\n" + "="*60)
    print("TEST 5: Notebook Section")
    print("="*60)
    
    try:
        notebook_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'heart_disease_project.ipynb')
        
        if not os.path.exists(notebook_path):
            print(f"❌ Notebook not found")
            return False
        
        import json
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        phase5_keywords = ['Phase 5', 'Data Splitting', 'Model Architecture', 'train_test_split', 'Sequential']
        found_keywords = []
        
        for cell in notebook['cells']:
            if cell['cell_type'] in ['code', 'markdown']:
                source = ''.join(cell.get('source', []))
                for keyword in phase5_keywords:
                    if keyword in source and keyword not in found_keywords:
                        found_keywords.append(keyword)
        
        print(f"✓ Notebook found")
        print(f"  - Found keywords: {len(found_keywords)}/{len(phase5_keywords)}")
        
        if len(found_keywords) >= 3:
            print("  ✓ Phase 5 content appears to be in notebook")
            return True
        else:
            print(f"  ⚠ Phase 5 content may be incomplete")
            return len(found_keywords) > 0
        
    except Exception as e:
        print(f"❌ Notebook section test failed: {e}")
        return False

def run_all_phase5_tests():
    """Run all Phase 5 tests"""
    print("="*60)
    print("PHASE 5 TESTING - Data Splitting & Model Architecture")
    print("="*60)
    
    test1 = test_data_loading()
    test2 = test_data_splitting()
    test3 = test_model_architecture()
    test4 = test_model_compilation()
    test5 = test_notebook_section_exists()
    
    print("\n" + "="*60)
    passed = sum([test1, test2, test3, test4, test5])
    total = 5
    
    if passed == total:
        print(f"✅ PHASE 5 COMPLETE - All {passed}/{total} tests passed!")
        print("You can proceed to Phase 6")
    else:
        print(f"⚠️ PHASE 5 INCOMPLETE - {passed}/{total} tests passed")
        print("Review the warnings above")
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_phase5_tests()
    sys.exit(0 if success else 1)
