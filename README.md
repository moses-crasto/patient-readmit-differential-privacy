# 🔐 Predicting Patient Readmission with Differential Privacy

This project explores the use of *differentially private machine learning* to predict hospital *patient readmissions* using structured healthcare data. It compares models trained *with and without differential privacy, demonstrating the trade-offs between **accuracy and data privacy*.

---

## 📁 Dataset

- Rows: ~181,284 patients
- Features: 46 columns including:
  - 🔢 *37 numeric features* (lab results, vital stats)
  - 🏷 *6 categorical features* (diagnosis codes, demographics)
  - 🎯 *Target column*: readmission (0 = No, 1 = Yes)

> *Note*: The dataset includes both raw clinical variables and lab test pivots. Leakage columns (next_admission_date, days_to_readmission) are removed.

---

## 🔍 Problem Statement

> *Can we accurately predict if a patient will be readmitted, while preserving privacy using Differential Privacy (DP)?*

We implement both:
- ✅ *Standard neural network model (no noise)*
- 🔒 **DP-SGD model using tensorflow-privacy**

---

## 🧪 Methods

### 1. *Preprocessing*
- Standardized numerical features (StandardScaler)
- One-hot encoded categorical features (OneHotEncoder)
- Combined using ColumnTransformer
- Result: Sparse matrix with ~100k features

### 2. *Model Architecture*
- Input layer (sparse)
- Dense (128 → 64 → 1)
- Activation: ReLU, Sigmoid

### 3. *Differential Privacy*
- Optimizer: DPKerasAdamOptimizer
- Key hyperparameters:
  - noise_multiplier: 0.7
  - l2_norm_clip: 1.0
  - batch_size: 128, 256, 512
- Privacy guarantee: (ε, δ), where δ = 1e-5

---

## 📊 Results

We compare models on:
- ✅ *Accuracy*
- 🔐 *Privacy Budget (ε)*

---

## 🖼 Visualizations

- Accuracy vs Batch Size  
- Privacy ε vs Batch Size  

---

📌 Future Work
Integrate synthetic data generation for improved utility

Extend to multi-class prediction for specific conditions

Deploy model with a secure API (e.g., FastAPI + DP inference)
