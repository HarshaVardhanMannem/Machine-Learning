# 🤖 Machine Learning Concepts — Hands-On Experiments

> A rigorous collection of Jupyter notebooks exploring foundational and intermediate Machine Learning concepts through real-world datasets, controlled experiments, and visualized results.

---

## 📌 Overview

This repository contains carefully designed experiments that go beyond textbook definitions. Each notebook is built to **demonstrate the *why* behind each concept**, using real datasets, side-by-side comparisons, and performance metrics.

Whether you're a practitioner brushing up on fundamentals, a student building intuition, or a recruiter evaluating technical depth — this collection highlights the engineering discipline required to build reliable ML systems.

---

## 🗂️ Experiments

### 1. 🖥️ CPU vs. GPU Performance Benchmark
**Notebook:** [`ML concepts/CPU_vs_GPU.ipynb`](ML%20concepts/CPU_vs_GPU.ipynb)

**Problem:** How much faster is a GPU compared to a CPU for the core computation in deep learning — matrix multiplication?

**Approach:**
- Benchmarked `torch.matmul()` across **5 matrix sizes**: 256×256 → 4096×4096
- Used `torch.cuda.synchronize()` for accurate GPU timing (eliminates async measurement bias)
- Performed a GPU warm-up run to exclude CUDA context initialization overhead

**Key Findings:**
- GPU speedup grows **non-linearly** with matrix size, reflecting the advantage of massive parallelism at scale
- At 4096×4096, the GPU completes the operation orders of magnitude faster than the CPU
- Results are visualized as a comparative bar chart showing CPU vs. GPU time per matrix size

**Why it matters:** Neural network training is dominated by matrix multiplications. Understanding this gap directly explains why GPU/TPU hardware is non-negotiable for production-scale deep learning.

---

### 2. 📏 Standardization in Machine Learning
**Notebook:** [`ML concepts/Standardization in ML.ipynb`](ML%20concepts/Standardization%20in%20ML.ipynb)

**Dataset:** Social Network Ads (Age, EstimatedSalary → Purchase prediction)

**Problem:** Does feature scaling via standardization improve model accuracy? Does it affect all model types equally?

**Approach:**
- Applied `StandardScaler` (zero mean, unit variance) fit **only on training data** to prevent data leakage
- Trained both **Logistic Regression** and **Decision Tree** classifiers on raw vs. scaled features
- Visualized feature distributions before/after scaling using KDE plots and scatter plots

**Key Engineering Decisions:**
- Scaler fitted exclusively on `X_train`, then applied to both train and test sets — a critical detail that prevents information leakage from the test set
- Compared models with inherently different sensitivity to scale (distance-based vs. tree-based)

**Key Findings:**
- Logistic Regression accuracy improves significantly after standardization (gradient descent converges faster and more reliably on scaled features)
- Decision Tree accuracy is **unaffected** by scaling — as expected, since tree splits are invariant to monotonic feature transformations
- KDE plots visually confirm that standardization preserves the shape of the distribution while re-centering it to μ=0, σ=1

---

### 3. 🔄 Normalization in Machine Learning
**Notebook:** [`ML concepts/Normalization in ML.ipynb`](ML%20concepts/Normalization%20in%20ML.ipynb)

**Dataset:** Wine dataset (Alcohol, Malic Acid → Class label)

**Problem:** When should you use Min-Max Normalization instead of Standardization, and what does it actually do to your data?

**Approach:**
- Applied `MinMaxScaler` to compress all feature values into the [0, 1] range
- Visualized feature distributions and scatter plots before and after transformation using KDE plots
- Demonstrated class separability in both feature spaces

**Key Engineering Decisions:**
- Used a multi-class (3-class) dataset to show normalization's effect across overlapping clusters
- Compared raw vs. normalized distributions for features with very different scales (Alcohol: ~11–15, Malic Acid: ~0.7–5.8)

**Key Findings:**
- After normalization, features with very different raw scales are brought onto a comparable range without distorting their relative relationships
- The shape of each feature's distribution is **preserved** — unlike standardization, normalization doesn't shift the distribution to zero-mean
- Scatter plot class clusters remain clearly separable after scaling, confirming the transformation doesn't destroy discriminative structure

---

### 4. 🎯 Classification Metrics: Accuracy, Precision, Recall & F1 Score
**Notebook:** [`ML concepts/Accuracy_Precision_Recall_F1 score  in ML Demo.ipynb`](ML%20concepts/Accuracy_Precision_Recall_F1%20score%20%20in%20ML%20Demo.ipynb)

**Dataset:** Breast Cancer Wisconsin (30 features → Malignant/Benign classification)

**Problem:** Accuracy alone is insufficient for evaluating classifiers in imbalanced or high-stakes settings. How do we interpret the full suite of classification metrics?

**Approach:**
- Trained a `LogisticRegression` model with stratified train/test split (70/30)
- Computed full confusion matrix, accuracy, precision, recall, and F1 score
- Generated **Precision-Recall curve** and **ROC curve** with AUC score

**Results Achieved:**

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 94.7%  |
| Precision | 93.8%  |
| Recall    | 98.1%  |
| F1 Score  | 95.9%  |
| AUC (ROC) | ~0.99  |

**Key Insights:**
- **High Recall (98.1%)** is the critical metric here — in cancer screening, missing a positive (false negative) is far more dangerous than a false alarm
- The small precision-recall gap reflects a deliberate bias toward sensitivity over specificity — the correct trade-off in medical diagnostics
- The near-perfect AUC demonstrates that the model's **probability calibration** is excellent across all classification thresholds

---

## 🧠 Core ML Concepts Covered

| Concept | Notebook |
|---|---|
| Hardware acceleration (CPU vs. GPU) | CPU vs. GPU Benchmark |
| Feature scaling — standardization | Standardization in ML |
| Feature scaling — normalization | Normalization in ML |
| Data leakage prevention | Standardization & Normalization |
| Classification metrics (Accuracy, Precision, Recall, F1) | Classification Metrics Demo |
| Confusion matrix interpretation | Classification Metrics Demo |
| ROC curve & AUC | Classification Metrics Demo |
| Precision-Recall tradeoffs | Classification Metrics Demo |
| Model comparison (Logistic Regression vs. Decision Tree) | Standardization in ML |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3** | Core language |
| **PyTorch** | GPU/CPU benchmarking |
| **scikit-learn** | ML models, preprocessing, metrics |
| **NumPy / Pandas** | Data manipulation |
| **Matplotlib / Seaborn** | Visualization |
| **Jupyter Notebook** | Interactive experimentation |

---

## 📁 Repository Structure

```
Machine-Learning/
└── ML concepts/
    ├── data/
    │   ├── Social_Network_Ads.csv       # Social network purchase prediction dataset
    │   └── wine_data.csv                # Wine classification dataset
    ├── CPU_vs_GPU.ipynb                  # CPU vs. GPU matrix multiplication benchmark
    ├── Standardization in ML.ipynb      # StandardScaler experiment with model comparison
    ├── Normalization in ML.ipynb        # MinMaxScaler experiment with wine dataset
    └── Accuracy_Precision_Recall_F1 score  in ML Demo.ipynb  # Full classification metrics demo
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch jupyter
```

### Run Notebooks

```bash
git clone https://github.com/HarshaVardhanMannem/Machine-Learning.git
cd Machine-Learning
jupyter notebook
```

Navigate to `ML concepts/` and open any notebook to explore.

> **Note:** The CPU vs. GPU benchmark requires a CUDA-compatible GPU for GPU timing. It will still run on CPU-only machines and report CPU timings.

---

## 💡 Engineering Principles Demonstrated

- **No data leakage** — scalers are always fit on training data only and applied to test data
- **Fair benchmarking** — GPU warm-up and `cuda.synchronize()` used for accurate timing
- **Model-specific analysis** — scale-sensitive (Logistic Regression) vs. scale-invariant (Decision Tree) models tested separately
- **Metric selection by context** — recall prioritized over accuracy in the medical classification task
- **Visual validation** — all transformations and results verified through distribution plots and scatter plots

---

## 👤 Author

**Harsha Vardhan Mannem**  
[GitHub](https://github.com/HarshaVardhanMannem)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
