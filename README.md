# Heart Disease Prediction - Deep Learning Project

## 📋 Project Overview

This project implements a deep learning model to predict heart disease using patient medical data. The project is divided into three parts:

- **Person A (Completed):** Data preparation, EDA, and preprocessing pipeline
- **Person B:** Model building and training
- **Person C:** Evaluation and reporting

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- VS Code (recommended)
- Git (for version control)

### Environment Setup

1. **Navigate to project directory:**
   ```powershell
   cd heart_disease_dl_project
   ```

2. **Activate virtual environment:**
   ```powershell
   .venv\Scripts\activate
   ```
   You should see `(.venv)` in your terminal prompt.

3. **Install dependencies (if not already installed):**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Select Python interpreter in VS Code:**
   - Press `Ctrl+Shift+P`
   - Type: "Python: Select Interpreter"
   - Choose: `.venv\Scripts\python.exe`

---

## 🔧 Git Setup (Optional but Recommended)

If you want to use Git for version control:

1. **Initialize Git repository (already done):**
   ```powershell
   git init
   ```

2. **Add files to Git:**
   ```powershell
   git add .
   ```

3. **Make initial commit:**
   ```powershell
   git commit -m "Initial commit: Person A work complete"
   ```

4. **Create a repository on GitHub/GitLab and push:**
   ```powershell
   git remote add origin <your-repo-url>
   git branch -M main
   git push -u origin main
   ```

**Note:** The `.venv/` folder is excluded from Git (see `.gitignore`)

---

## 📁 Project Structure

```
heart_disease_dl_project/
├── data/
│   └── heart.csv                    # Dataset (918 patients)
├── notebooks/
│   └── heart_disease_project.ipynb  # Main analysis notebook
├── models/                          # Saved models (for Person B)
├── report/                          # Final report (for Person C)
├── tests/                           # Test scripts
│   ├── test_phase1_setup.py
│   ├── test_phase2_data_loading.py
│   ├── test_phase3_eda.py
│   └── test_phase4_preprocessing.py
├── .venv/                           # Virtual environment
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🏃 How to Run

### Running the Notebook

1. **Open VS Code** in the project directory:
   ```powershell
   code .
   ```

2. **Open the notebook:**
   - Navigate to `notebooks/heart_disease_project.ipynb`
   - Make sure the kernel is selected (top-right corner)
   - Choose the `.venv` Python interpreter

3. **Run cells:**
   - Run individual cells: `Shift + Enter`
   - Run all cells: `Ctrl + Shift + P` → "Run All Cells"

### Using the Preprocessing Function

Person B can use the preprocessing function directly:

```python
# In a new cell or Python script
# The function is already defined in the notebook Section 5
# Just run the cells in Section 5, then call:

X, y, feature_names, scaler = load_and_preprocess()

# Now ready for model training
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

### Running Tests

To verify everything works:

```powershell
# Test Phase 1 (Environment setup)
python tests/test_phase1_setup.py

# Test Phase 2 (Data loading)
python tests/test_phase2_data_loading.py

# Test Phase 3 (EDA)
python tests/test_phase3_eda.py

# Test Phase 4 (Preprocessing)
python tests/test_phase4_preprocessing.py
```

---

## 📊 What Was Done (Person A - Completed)

### Phase 1: Environment Setup ✅
- Created project folder structure
- Set up Python virtual environment
- Installed required libraries (pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter, tensorflow)
- Created main Jupyter notebook

### Phase 2: Data Loading & Inspection ✅
- Loaded heart disease dataset (918 patients, 12 features)
- Performed basic data inspection
- Documented dataset characteristics
- Verified data quality

### Phase 3: Exploratory Data Analysis (EDA) ✅
- Analyzed missing values (none found)
- Analyzed target variable distribution (balanced: 44.7% vs 55.3%)
- Visualized numeric features (histograms, box plots)
- Visualized categorical features (bar charts)
- Performed correlation analysis
- Identified key predictors: Oldpeak (0.404), MaxHR (0.400), Age (0.282)

### Phase 4: Data Preprocessing Pipeline ✅
- Handled missing values (none found, but code handles them)
- Separated features and target
- Encoded categorical variables (one-hot encoding: 11 → 15 features)
- Scaled all features using StandardScaler (mean=0, std=1)
- Created `load_and_preprocess()` function for Person B
- Verified preprocessing quality

### Deliverable for Person B

**Function:** `load_and_preprocess()`

**Returns:**
- `X_scaled`: Scaled feature matrix (918, 15) - numpy array
- `y`: Target labels (918,) - numpy array  
- `feature_names`: List of 15 feature names
- `scaler`: Fitted StandardScaler object

**Usage:**
```python
X, y, feature_names, scaler = load_and_preprocess()
```

---

## 📦 Dataset Information

- **Source:** Kaggle - Heart Failure Prediction Dataset
- **Size:** 918 patients
- **Features:** 11 input features + 1 target
- **Target:** HeartDisease (0 = No disease, 1 = Disease)
- **Class Distribution:** 44.7% No Disease, 55.3% Disease (balanced)

**Features:**
- **Numeric:** Age, RestingBP, Cholesterol, MaxHR, Oldpeak, FastingBS
- **Categorical:** Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope

---

## 🔧 Dependencies

All required packages are listed in `requirements.txt`:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
- tensorflow

Install with:
```powershell
pip install -r requirements.txt
```

---

## ⚠️ Known Issues

**TensorFlow DLL Loading (Windows):**
- TensorFlow may show DLL loading errors on Windows
- This doesn't affect Phases 2-4 (data work)
- Person B will need TensorFlow for model training
- **Solution:** Install Visual C++ Redistributables: https://aka.ms/vs/17/release/vc_redist.x64.exe

---

## 📝 Next Steps for Person B

1. Open `notebooks/heart_disease_project.ipynb`
2. Call `load_and_preprocess()` to get clean data
3. Split data into train/validation/test sets (70/15/15)
4. Build neural network model
5. Train with early stopping
6. Perform hyperparameter tuning
7. Save model to `models/heart_failure_model.h5`

---

## 🧪 Testing

All phases have been tested and verified:

- ✅ Phase 1: Environment setup
- ✅ Phase 2: Data loading
- ✅ Phase 3: EDA completion
- ✅ Phase 4: Preprocessing pipeline

Run tests to verify:
```powershell
python tests/test_phase4_preprocessing.py
```

---

## 📞 Support

If you encounter issues:

1. **Virtual environment not activating:**
   - Make sure you're in the project directory
   - Try: `.venv\Scripts\activate.ps1`

2. **Notebook kernel not found:**
   - Select Python interpreter in VS Code (top-right)
   - Choose `.venv\Scripts\python.exe`

3. **Import errors:**
   - Activate virtual environment
   - Reinstall: `pip install -r requirements.txt`

---

## 📄 License

This project is part of a team assignment for educational purposes.

---

**Project Status:** Person A work complete ✅  
**Ready for:** Person B (Model & Training)
