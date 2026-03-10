# 📚 ML Concepts — Jupyter Notebook Experiments

> Hands-on Jupyter notebooks covering foundational and intermediate Machine Learning concepts. Each experiment is built around a real dataset with side-by-side comparisons and visual results to illustrate *why* each concept matters.

---

## 🗂️ Notebooks

### 1. 🖥️ CPU vs. GPU Performance Benchmark
**File:** [`CPU_vs_GPU.ipynb`](CPU_vs_GPU.ipynb)

**Problem:** How much faster is a GPU compared to a CPU for the core computation in deep learning — matrix multiplication?

**Approach:**
- Benchmarked `torch.matmul()` across **5 matrix sizes**: 256×256 → 4096×4096
- Used `torch.cuda.synchronize()` for accurate GPU timing (eliminates async measurement bias)
- Performed a GPU warm-up run to exclude CUDA context initialization overhead

**Key Findings:**
- GPU speedup grows **non-linearly** with matrix size, reflecting massive-parallelism advantages at scale
- At 4096×4096, the GPU completes the operation orders of magnitude faster than the CPU
- Results visualised as a comparative bar chart showing CPU vs. GPU time per matrix size

**Why it matters:** Neural network training is dominated by matrix multiplications. Understanding this gap directly explains why GPU/TPU hardware is essential for production-scale deep learning.

---

### 2. 📏 Standardization in Machine Learning
**File:** [`Standardization in ML.ipynb`](Standardization%20in%20ML.ipynb)

**Dataset:** Social Network Ads (Age, EstimatedSalary → Purchase prediction)

**Problem:** Does feature scaling via standardization improve model accuracy? Does it affect all model types equally?

**Approach:**
- Applied `StandardScaler` (zero mean, unit variance) fit **only on training data** to prevent data leakage
- Trained both **Logistic Regression** and **Decision Tree** classifiers on raw vs. scaled features
- Visualised feature distributions before/after scaling using KDE plots and scatter plots

**Key Findings:**
- Logistic Regression accuracy improves significantly after standardization
- Decision Tree accuracy is **unaffected** by scaling — tree splits are invariant to monotonic feature transformations
- KDE plots confirm that standardization preserves distribution shape while re-centering to μ=0, σ=1

---

### 3. 🔄 Normalization in Machine Learning
**File:** [`Normalization in ML.ipynb`](Normalization%20in%20ML.ipynb)

**Dataset:** Wine dataset (Alcohol, Malic Acid → Class label)

**Problem:** When should you use Min-Max Normalization instead of Standardization, and what does it actually do to your data?

**Approach:**
- Applied `MinMaxScaler` to compress all feature values into the [0, 1] range
- Visualised feature distributions and scatter plots before and after transformation using KDE plots
- Demonstrated class separability in both feature spaces

**Key Findings:**
- Features with very different raw scales are brought onto a comparable range without distorting their relative relationships
- The shape of each feature's distribution is **preserved** — unlike standardization, normalization doesn't shift the distribution to zero-mean
- Scatter plot class clusters remain clearly separable after scaling

---

### 4. 🎯 Classification Metrics: Accuracy, Precision, Recall & F1 Score
**File:** [`Accuracy_Precision_Recall_F1 score  in ML Demo.ipynb`](Accuracy_Precision_Recall_F1%20score%20%20in%20ML%20Demo.ipynb)

**Dataset:** Breast Cancer Wisconsin (30 features → Malignant/Benign classification)

**Problem:** Accuracy alone is insufficient for evaluating classifiers in imbalanced or high-stakes settings. How do we interpret the full suite of classification metrics?

**Approach:**
- Trained a `LogisticRegression` model with stratified train/test split (70/30)
- Computed full confusion matrix, accuracy, precision, recall, and F1 score
- Generated **Precision-Recall curve** and **ROC curve** with AUC score

**Results:**

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 94.7%  |
| Precision | 93.8%  |
| Recall    | 98.1%  |
| F1 Score  | 95.9%  |
| AUC (ROC) | ~0.99  |

**Key Insights:**
- **High Recall (98.1%)** is the critical metric — in cancer screening, missing a positive (false negative) is far more dangerous than a false alarm
- Near-perfect AUC demonstrates excellent model probability calibration across all classification thresholds

---

### 5. 📐 Euclidean vs. Manhattan Distance in Machine Learning
**File:** [`Euclidean_vs_Manhattan_Distance_ML.ipynb`](Euclidean_vs_Manhattan_Distance_ML.ipynb)

**Dataset:** Synthetic 2D dataset (300 samples, 3 clusters, generated with `sklearn.datasets.make_blobs`)

**Problem:** How does the choice of distance metric (Euclidean vs. Manhattan) affect the behaviour of ML algorithms like KNN, K-Means, and DBSCAN?

**Approach:**
- Visualised geometric intuition: shortest straight-line path (Euclidean) vs. grid-path distance (Manhattan)
- Compared **KNN** classification decision boundaries under both metrics
- Compared **K-Means** (Euclidean) vs. **K-Medians** (Manhattan) clustering shapes
- Analysed outlier sensitivity and high-dimensional behaviour of each metric
- Provided a practical decision matrix for choosing the right metric

**Key Findings:**
- Euclidean distance is better suited to continuous geometric data and low-dimensional spaces
- Manhattan distance is more robust to outliers and performs better on sparse or high-dimensional data
- Decision boundary shapes and cluster assignments differ meaningfully between the two metrics
- Practical guidance: use Euclidean for image/spatial data; use Manhattan for text, NLP, or grid-structured data

---

## 🧠 Concepts at a Glance

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
| Distance metrics (Euclidean vs. Manhattan) | Euclidean vs. Manhattan Distance |
| KNN, K-Means, K-Medians, DBSCAN | Euclidean vs. Manhattan Distance |
| Outlier sensitivity analysis | Euclidean vs. Manhattan Distance |

---

## 📁 Folder Structure

```
ML concepts/
├── data/
│   ├── Social_Network_Ads.csv           # Purchase prediction dataset
│   └── wine_data.csv                    # Wine classification dataset
├── CPU_vs_GPU.ipynb                     # CPU vs. GPU benchmark
├── Standardization in ML.ipynb         # StandardScaler experiment
├── Normalization in ML.ipynb           # MinMaxScaler experiment
├── Accuracy_Precision_Recall_F1 score  in ML Demo.ipynb  # Classification metrics
└── Euclidean_vs_Manhattan_Distance_ML.ipynb              # Distance metrics comparison
```

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

## 🚀 Getting Started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch jupyter
jupyter notebook
```

Open any notebook in this folder to explore the experiments interactively.

> **Note:** The CPU vs. GPU benchmark requires a CUDA-compatible GPU for GPU timing. It will still run on CPU-only machines and report CPU timings only.

---

## 💡 Engineering Principles Demonstrated

- **No data leakage** — scalers are always fit on training data only and applied to test data
- **Fair benchmarking** — GPU warm-up and `cuda.synchronize()` used for accurate timing
- **Model-specific analysis** — scale-sensitive vs. scale-invariant models tested separately
- **Metric selection by context** — recall prioritized over accuracy in the medical classification task
- **Visual validation** — all transformations and results verified through distribution plots and scatter plots
