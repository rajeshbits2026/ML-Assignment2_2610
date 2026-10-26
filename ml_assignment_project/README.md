# Machine Learning Assignment 2 — Dry Bean Classification

## 1. Problem Statement

The objective of this project is to build and compare five supervised machine-learning classification algorithms on the Dry Bean dataset. The models classify dry bean varieties using numerical characteristics extracted from bean images.

The algorithms evaluated are:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbours (KNN)
- Naive Bayes
- Support Vector Machine (SVM)

Each model is evaluated using Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Official source:** https://archive.ics.uci.edu/dataset/602/dry%2Bbean

The dataset contains 13,611 instances and 16 numerical input features extracted from images of dry beans. The target variable, `Class`, identifies one of seven varieties: Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira.

The dataset is suitable for multiclass classification and meets the assignment's minimum requirements for instance count and number of input features. Data is divided into 80% for training and 20% for testing.

## 3. Methodology

1. Load the dataset and separate the input features from the target variable.
2. Divide the data into training and testing subsets using a stratified 80:20 split with `random_state=42`.
3. Train five classification algorithms using the same training split.
4. Evaluate each trained model on the held-out test data.
5. Save the trained models and evaluation results.
6. Use the Streamlit application to select a model, upload labelled test data and inspect its evaluation results.

The saved models are loaded by the application. Uploading a test file does not retrain them.

## 4. Model Performance Comparison

The following results were obtained in the current experiment. Values are rounded to four decimal places.

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9948 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.8920 | 0.9450 | 0.8917 | 0.8920 | 0.8916 | 0.8696 |
| KNN | 0.9166 | 0.9833 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9672 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| SVM | **0.9221** | **0.9949** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

AUC is calculated using macro one-vs-rest evaluation. Precision, Recall and F1-score use weighted averaging. MCC is the multiclass Matthews Correlation Coefficient.

## 5. Observations and Discussion

**Logistic Regression:** The model achieved 92.14% accuracy and an AUC of 0.9948. Its results are close to those of SVM, indicating that a relatively simple linear classification model performs strongly on this dataset.

**Decision Tree:** The model achieved 89.20% accuracy. Its results were lower than those of Logistic Regression, KNN and SVM. A single tree may not capture the dataset's class boundaries as effectively as the better-performing alternatives in this experiment.

**K-Nearest Neighbours:** KNN achieved 91.66% accuracy and an MCC of 0.8992. Its performance was competitive with the leading models, although it did not achieve the highest score on the reported metrics.

**Naive Bayes:** The model achieved 76.39% accuracy, the lowest among the five models. Its AUC was nevertheless 0.9672. This difference shows that AUC and accuracy measure different aspects of classification performance. The model's probability rankings can be informative even when its final class predictions are less accurate.

**Support Vector Machine:** SVM achieved the highest reported accuracy (92.21%), AUC (0.9949), weighted precision (0.9224), weighted recall (0.9221), weighted F1-score (0.9222) and MCC (0.9059). It is the strongest overall performer on this test split, although its advantage over Logistic Regression is small.

## 6. Conclusion

On the basis of the reported held-out test results, SVM is the best-performing model among the five evaluated algorithms. Logistic Regression is a close second, while KNN also demonstrates competitive performance. Decision Tree and Naive Bayes produce lower accuracy in this experiment.

These conclusions apply to the selected dataset, preprocessing, split and model configurations. Further validation, including cross-validation and class-wise error analysis, could provide additional evidence about how well the models generalise.

## 7. Streamlit Application

The application allows a user to:

- Upload a labelled CSV test dataset.
- Select one of the five saved classification models.
- View Accuracy, AUC, Precision, Recall, F1-score and MCC.
- Inspect a confusion matrix and classification report.
- Compare model performance.

The uploaded CSV must contain the same input feature columns used during training and a labelled target column named `Class`.

## 8. Repository Contents

- `app.py` — Streamlit application.
- `requirements.txt` — Python dependencies.
- `ML_Assignment_2_End_to_End.ipynb` — notebook containing the project workflow.
- `test_data.csv` — labelled test dataset.
- `model_comparison.csv` — model evaluation results.
- `metadata.json` — project metadata.
- `model/` — saved model files for all five algorithms.

## 9. How to Run Locally


(a). Install packages from `requirements.txt`.

From the project directory, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

(b). Run the notebook from top to bottom to train models and create the project artifacts.

(c). Start the application:

```bash
python -m streamlit run app.py
```
(d).  Open the local URL displayed in the terminal, upload `test_data.csv`, select a model and review its results.

## 10. Links

- **GitHub repository:** [Add the actual repository URL]
- **Live Streamlit application:** [Add the actual deployed application URL]
- **Dataset source:** https://archive.ics.uci.edu/dataset/602/dry%2Bbean










## How to run
1. Install packages from `requirements.txt`.
2. Run the notebook from top to bottom to train models and create the project artifacts.
3. From this folder, run `streamlit run app.py`.
4. Upload a labelled CSV with the same feature columns and target column.

## Repository contents
- `app.py` — Streamlit evaluation interface
- `requirements.txt` — dependencies
- `README.md` — project documentation
- `test_data.csv` — labelled held-out test data
- `model/` — saved trained pipelines
- `metadata.json` — feature, target, class, and model-file metadata
- `model_comparison.csv` — measured comparison results

## Important limitations
The app evaluates labelled test data. It does not train or update models on uploaded data. Metric values can change if you change the dataset version, split, or model settings.
