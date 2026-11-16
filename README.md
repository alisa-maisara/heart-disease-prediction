# Heart Disease Prediction with a Neural Network

> **Disclaimer:** This is an academic coursework project created for learning machine learning concepts. It is **not** a medical tool and must not be used to diagnose or make decisions about anyone's health.

A neural network classifier (`MLPClassifier`) that predicts whether a patient has heart disease based on 13 clinical health measurements. Built as a group project (team of 5) for **CSC583: Artificial Intelligence Algorithms** at Universiti Teknologi MARA (UiTM) from March to August 2025.

---

## Dataset

A public heart disease dataset provided by our lecturer and available on Kaggle. The CSV file is not included in this repository.

* **Size:** 1,025 records, 13 input features, and 1 binary target (`0` = no heart disease, `1` = heart disease)
* **Features:** `age`, `sex`, `chest pain type`, `resting blood pressure`, `cholesterol`, `fasting blood sugar`, `resting ECG`, `maximum heart rate`, `exercise-induced angina`, `oldpeak`, `slope`, `number of vessels coloured by fluoroscopy`, and `thalassemia`
* **Data Quality:** No missing values

---

## Methodology & Workflow

1. **Data Preparation:** Checked for missing values, one-hot encoded categorical features using scikit-learn's `ColumnTransformer` and `OneHotEncoder` (expanding to 30 columns), and split the data 80/20 into training and test sets (`random_state=42`).
2. **Model Building:** Built a baseline `MLPClassifier` with one hidden layer of 100 neurons (`max_iter=500`).
3. **Model Training:** The baseline model converged in 113 iterations, verified using a training loss curve.
4. **Model Evaluation:** Evaluated test predictions using accuracy, precision, recall, F1-score, a confusion matrix, a classification report, and an ROC curve.
5. **Model Optimisation:** Ran five experiments varying hidden layer architecture, learning rate, and activation function.

---

## Results

### Baseline Model (Test Set: 205 Records)

| Metric | Score |
| :--- | :---: |
| **Accuracy** | 0.83 |
| **Precision** | 0.78 |
| **Recall** | 0.91 |
| **F1-Score** | 0.84 |
| **ROC AUC** | 0.92 |

* **Confusion Matrix:** 76 True Negatives, 26 False Positives, 9 False Negatives, and 94 True Positives.
* **Clinical Trade-off Interpretation:** The baseline catches most positive cases (high recall of `0.91`) while flagging some healthy patients (lower precision of `0.78`). In healthcare screening, this trade-off matters because a missed positive case (false negative) and a false alarm (false positive) carry very different consequences.

### Optimisation Experiments

| Exp # | Architecture | Learning Rate | Activation | Accuracy | Precision | Recall | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `(100,)` | `0.001` | `relu` | 0.829 | 0.783 | 0.913 | 0.843 |
| **2** | `(100, 50)` | `0.001` | `relu` | 0.805 | 0.794 | 0.825 | 0.810 |
| **3** | `(200,)` | `0.001` | `relu` | 0.810 | 0.814 | 0.806 | 0.810 |
| **4** | `(100,)` | `0.01` | `relu` | 0.790 | 0.714 | **0.971** | 0.823 |
| **5** | `(100,)` | `0.001` | `tanh` | **0.883** | **0.838** | 0.951 | **0.891** |

Experiment 5 (`tanh` activation) achieved the strongest overall balance across accuracy (`88.3%`) and F1-score (`0.891`).

> **Methodological Limitation:** All five experiments were compared directly against the same test set to select the best configuration, meaning Experiment 5's metrics may be slightly optimistic. Using k-fold cross-validation or a separate validation split would provide a more unbiased performance estimate. Additionally, the dataset is relatively small (1,025 rows), which limits generalizability.

---

## My Contribution

I led **Model Evaluation (Step 4)**: computing evaluation metrics, generating the confusion matrix, classification report, and ROC curve, and interpreting the clinical implications of precision vs. recall.

---

## Tech Stack

* **Language:** Python
* **Libraries:** `scikit-learn`, `pandas`, `matplotlib`
* **Environment:** Google Colab

---

## How to Run the Notebook

1. You Download the dataset from Kaggle and save it as `HeartDiseaseTrain-Test.csv`.
2. Open the notebook (`heart_disease.ipynb`) in Google Colab or Jupyter Notebook.
3. Upload `HeartDiseaseTrain-Test.csv` to your working directory (or update the file path in the first code cell).
4. Run all cells sequentially.

---

## Team & Status

* **Team:** Group of 5 Computer Science students, Universiti Teknologi MARA (UiTM)
* **Status:** Completed (Coursework — Semester 3, 2025)
