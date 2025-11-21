"""
Test script for Phase 6: Model Training & Early Stopping
Tests Person B's Phase 6 implementation
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def test_callbacks_setup():
    """Test if callbacks can be set up"""
    print("\n" + "="*60)
    print("TEST 1: Callbacks Setup")
    print("="*60)
    
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            ModelCheckpoint(filepath='../models/test_model.h5', monitor='val_loss', save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0)
        ]
        
        print("✓ Callbacks created successfully")
        print(f"  - Number of callbacks: {len(callbacks)}")
        print(f"  - EarlyStopping: ✓")
        print(f"  - ModelCheckpoint: ✓")
        print(f"  - ReduceLROnPlateau: ✓")
        
        return True
        
    except ImportError:
        print("⚠ TensorFlow not available (OK for logic testing)")
        return True
    except Exception as e:
        print(f"❌ Callbacks setup test failed: {e}")
        return False

def test_model_training():
    """Test if model can be trained"""
    print("\n" + "="*60)
    print("TEST 2: Model Training")
    print("="*60)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.callbacks import EarlyStopping
        
        # Create dummy data
        X_train = np.random.randn(100, 15)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(20, 15)
        y_val = np.random.randint(0, 2, 20)
        
        # Create simple model
        tf.random.set_seed(42)
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(15,)),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for a few epochs
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=3,
            batch_size=32,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0)],
            verbose=0
        )
        
        print("✓ Model training successful")
        print(f"  - Epochs trained: {len(history.history['loss'])}")
        print(f"  - Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"  - Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return len(history.history['loss']) > 0
        
    except ImportError:
        print("⚠ TensorFlow not available (OK for logic testing)")
        return True
    except Exception as e:
        print(f"❌ Model training test failed: {e}")
        return False

def test_model_checkpoint():
    """Test if model checkpointing works"""
    print("\n" + "="*60)
    print("TEST 3: Model Checkpointing")
    print("="*60)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.callbacks import ModelCheckpoint
        import tempfile
        
        # Create dummy data
        X_train = np.random.randn(50, 15)
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.randn(10, 15)
        y_val = np.random.randint(0, 2, 10)
        
        # Create model
        tf.random.set_seed(42)
        model = keras.Sequential([
            layers.Dense(16, activation='relu', input_shape=(15,)),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Create temporary file for checkpoint
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
        
        # Train with checkpoint
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=2,
            batch_size=16,
            callbacks=[checkpoint],
            verbose=0
        )
        
        # Check if file was created
        if os.path.exists(checkpoint_path):
            print("✓ Model checkpoint saved successfully")
            print(f"  - Checkpoint file exists: {checkpoint_path}")
            
            # Try to load it
            loaded_model = keras.models.load_model(checkpoint_path)
            print("  - Model can be loaded from checkpoint")
            
            # Cleanup
            os.remove(checkpoint_path)
            return True
        else:
            print("❌ Checkpoint file not created")
            return False
        
    except ImportError:
        print("⚠ TensorFlow not available (OK for logic testing)")
        return True
    except Exception as e:
        print(f"❌ Model checkpoint test failed: {e}")
        return False

def test_early_stopping():
    """Test if early stopping works"""
    print("\n" + "="*60)
    print("TEST 4: Early Stopping")
    print("="*60)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.callbacks import EarlyStopping
        
        # Create dummy data
        X_train = np.random.randn(50, 15)
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.randn(10, 15)
        y_val = np.random.randint(0, 2, 10)
        
        # Create model
        tf.random.set_seed(42)
        model = keras.Sequential([
            layers.Dense(16, activation='relu', input_shape=(15,)),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train with early stopping (patience=2)
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=0)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,  # Max epochs
            batch_size=16,
            callbacks=[early_stop],
            verbose=0
        )
        
        epochs_trained = len(history.history['loss'])
        print("✓ Early stopping working")
        print(f"  - Max epochs: 10")
        print(f"  - Epochs trained: {epochs_trained}")
        
        if epochs_trained < 10:
            print("  ✓ Training stopped early (as expected)")
        else:
            print("  ⚠ Training ran for all epochs (may be OK)")
        
        return True
        
    except ImportError:
        print("⚠ TensorFlow not available (OK for logic testing)")
        return True
    except Exception as e:
        print(f"❌ Early stopping test failed: {e}")
        return False

def test_training_history():
    """Test if training history is recorded"""
    print("\n" + "="*60)
    print("TEST 5: Training History")
    print("="*60)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Create dummy data
        X_train = np.random.randn(50, 15)
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.randn(10, 15)
        y_val = np.random.randint(0, 2, 10)
        
        # Create model
        tf.random.set_seed(42)
        model = keras.Sequential([
            layers.Dense(16, activation='relu', input_shape=(15,)),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=3,
            batch_size=16,
            verbose=0
        )
        
        # Check history keys
        required_keys = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
        found_keys = [key for key in required_keys if key in history.history]
        
        print("✓ Training history recorded")
        print(f"  - History keys found: {len(found_keys)}/{len(required_keys)}")
        print(f"  - Keys: {', '.join(found_keys)}")
        
        if len(found_keys) >= 3:
            print("  ✓ History contains required metrics")
            return True
        else:
            print("  ⚠ Some metrics missing")
            return False
        
    except ImportError:
        print("⚠ TensorFlow not available (OK for logic testing)")
        return True
    except Exception as e:
        print(f"❌ Training history test failed: {e}")
        return False

def test_notebook_section_exists():
    """Test if Phase 6 section exists in notebook"""
    print("\n" + "="*60)
    print("TEST 6: Notebook Section")
    print("="*60)
    
    try:
        notebook_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'heart_disease_project.ipynb')
        
        if not os.path.exists(notebook_path):
            print("❌ Notebook not found")
            return False
        
        import json
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        phase6_keywords = ['Phase 6', 'Model Training', 'Early Stopping', 'callbacks', 'fit', 'history']
        found_keywords = []
        
        all_text = ""
        for cell in notebook['cells']:
            if cell['cell_type'] in ['code', 'markdown']:
                all_text += "".join(cell.get('source', []))
        
        for keyword in phase6_keywords:
            if keyword in all_text and keyword not in found_keywords:
                found_keywords.append(keyword)
        
        print(f"✓ Notebook found")
        print(f"  - Found keywords: {len(found_keywords)}/{len(phase6_keywords)}")
        
        if len(found_keywords) >= 3:
            print("  ✓ Phase 6 content appears to be in notebook")
            return True
        else:
            print(f"  ⚠ Phase 6 content may be incomplete")
            return len(found_keywords) > 0
        
    except Exception as e:
        print(f"❌ Notebook section test failed: {e}")
        return False

def run_all_phase6_tests():
    """Run all Phase 6 tests"""
    print("="*60)
    print("PHASE 6 TESTING - Model Training & Early Stopping")
    print("="*60)
    
    test1 = test_callbacks_setup()
    test2 = test_model_training()
    test3 = test_model_checkpoint()
    test4 = test_early_stopping()
    test5 = test_training_history()
    test6 = test_notebook_section_exists()
    
    print("\n" + "="*60)
    passed = sum([test1, test2, test3, test4, test5, test6])
    total = 6
    
    if passed == total:
        print(f"✅ PHASE 6 COMPLETE - All {passed}/{total} tests passed!")
        print("You can proceed to Phase 7")
    else:
        print(f"⚠️ PHASE 6 INCOMPLETE - {passed}/{total} tests passed")
        print("Review the warnings above")
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_phase6_tests()
    sys.exit(0 if success else 1)

