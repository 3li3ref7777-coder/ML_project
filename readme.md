# 💧 Water Quality Analysis & Predictive Modeling (ML Capstone)

[![Open In Colab](https://colab.research.google.com/drive/1N4RTOuZ04hIRgi2cvKWm6A_YJocFYCK3?usp=sharing)]
[![GitHub Repository](https://github.com/3li3ref7777-coder/ML_project)]
[![Dataset Source](https://www.kaggle.com/datasets/eissaislam/waterquality)]

An end-to-end Machine Learning Capstone project applying **Regression**, **Classification**, and **Unsupervised Clustering** on real-world water quality measurements (`waterQuality1.csv`) to predict chemical concentrations, assess water potability, and uncover natural sample groupings.

---


---

## 📁 Repository Structure

```text
├── data/
│   └── waterQuality1.csv            # Water quality dataset (waterQuality1.csv)
├── notebooks/
│   └── project_ML.ipynb        # Full EDA, preprocessing, and ML modeling notebook
├── Water Quality ML Capstone Report.pdf          # Comprehensive 4-section analytical report
└── README.md                    # Project documentation and results overview
```

---

## 📊 Dataset Overview

* **Raw Dimensions:** `7,999` samples × `21` columns (`20` physicochemical/biological features + `1` binary target).
* **Cleaned Dimensions:** `7,996` samples × `21` columns (after removing `3` rows with malformed `#NUM!` target entries).
* **Target Variables:**
  * **Classification Target (`is_safe`):** `0` = Unsafe (`88.59%`, 7,084 samples), `1` = Safe (`11.41%`, 912 samples).
  * **Regression Target (`bacteria`):** Continuous chemical concentration ranging from `0.00` to `1.00`.
* **Features Used (20):** `aluminium`, `ammonia`, `arsenic`, `barium`, `cadmium`, `chloramine`, `chromium`, `copper`, `flouride`, `bacteria`, `viruses`, `lead`, `nitrates`, `nitrites`, `mercury`, `perchlorate`, `radium`, `selenium`, `silver`, `uranium`.

---

## 🛠️ Methodology & Pipeline

1. **Data Cleaning & Preprocessing:**
   * Coerced string anomalies (`#NUM!`) to `NaN` across numeric columns (`ammonia` and `is_safe`).
   * Dropped `3` records missing ground-truth `is_safe` labels and applied column mean imputation to remaining numeric features.
   * Standardized features using `StandardScaler` (fit strictly on training splits) for distance- and gradient-based models.
2. **Exploratory Data Analysis (EDA):**
   * Analyzed feature distributions and identified top correlations (`viruses` & `bacteria`: `r = 0.6185`; `chloramine` & `perchlorate`: `r = 0.5887`).
   * Identified a ~8:1 class imbalance in `is_safe`, motivating stratified train-test splitting (`stratify=y`).
3. **Machine Learning Modeling:**
   * **Regression:** Ordinary Least Squares (`LinearRegression`) predicting `bacteria`.
   * **Classification:** `LogisticRegression` vs. `DecisionTreeClassifier` (`gini` and `entropy`, `max_depth=7`) predicting `is_safe`.
   * **Clustering:** `KMeans` (`k = 2..19`) evaluated via the Elbow Method and Silhouette Score.

---

## 📈 Modeling Results Summary

### 1. Regression (`Target: perchlorate`)
| Model | MAE | MSE | RMSE | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | `0.1801` | `0.05094049` | `0.2257` | **`0.5337`** |

### 2. Classification (`Target: is_safe`)
| Model | Accuracy | Precision | Recall | Confusion Matrix `[TN, FP, FN, TP]` |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `90.87%` | `0.7143` | `0.3297` | `[1394, 24, 122, 60]` |
| **Decision Tree (`gini`, depth=7)** | `96.06%` | `0.8993` | `0.7363` | `[1403, 15, 48, 134]` |
| **Decision Tree (`entropy`, depth=7)** | **`96.69%`** | **`0.9161`** | **`0.7802`** | **`[1405, 13, 40, 142]`** |

### 3. Clustering (`Unsupervised K-Means`)
| Algorithm | Optimal $K$ | Best Silhouette Score | Key Insight |
| :--- | :---: | :---: | :--- |
| **K-Means Clustering** | `2` | `0.1724` | Separates low-aluminium baseline samples (`~ -0.5` scaled) from elevated-aluminium profiles. |

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone(https://github.com/3li3ref7777-coder/ML_project)
   cd notebooks
   ```
2. **Install required dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```
3. **Launch the notebook:**
   ```bash
   jupyter notebook notebooks/project_ML.ipynb
   ```

---

## 🔮 Future Improvements

* **Ensemble Methods:** Train **Random Forest** and **Gradient Boosting (XGBoost / LightGBM)** models to capture non-linear feature interactions in both `bacteria` regression and `is_safe` classification.
* **Imbalance Handling:** Apply **SMOTE** oversampling or class-weighted loss (`class_weight='balanced'`) to boost minority-class Recall.
* **Dimensionality Reduction:** Use **PCA** prior to K-Means clustering to reduce multi-feature noise and improve cluster separation.
