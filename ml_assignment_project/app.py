
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Classifier Comparison", page_icon="📊", layout="wide")
st.title("Machine Learning Classifier Comparison")
st.write("Upload labelled test data to evaluate the saved classifiers. Models are loaded from disk; no retraining occurs.")

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
with open(ROOT / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

MODEL_FILES = metadata["model_filenames"]
FEATURES = metadata["feature_columns"]
TARGET = metadata["target_column"]
CLASS_NAMES = metadata["class_names"]
LABEL_TO_INT = {label: i for i, label in enumerate(CLASS_NAMES)}

@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()}

models = load_models()
uploaded = st.file_uploader("Upload a labelled test CSV", type=["csv"])
if uploaded is not None:
    data = pd.read_csv(uploaded)
    missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()
    if data.empty:
        st.error("The uploaded CSV contains no rows.")
        st.stop()

    unknown_labels = sorted(set(data[TARGET].astype(str)) - set(CLASS_NAMES))
    if unknown_labels:
        st.error(f"Target column contains unknown class labels: {unknown_labels}")
        st.stop()

    X_eval = data[FEATURES].copy()
    y_eval = data[TARGET].astype(str).map(LABEL_TO_INT).to_numpy()
    selected = st.selectbox("Select a classifier", ["Compare all models"] + list(models.keys()))

    def auc_score(model, X, y):
        try:
            scores = model.predict_proba(X)
            if len(CLASS_NAMES) == 2:
                return roc_auc_score(y, scores[:, 1])
            return roc_auc_score(y, scores, multi_class="ovr", average="macro",
                                 labels=np.arange(len(CLASS_NAMES)))
        except (ValueError, TypeError, AttributeError):
            return float("nan")

    def evaluate(name, model):
        pred = model.predict(X_eval)
        return {
            "Model": name,
            "Accuracy": accuracy_score(y_eval, pred),
            "AUC": auc_score(model, X_eval, y_eval),
            "Precision": precision_score(y_eval, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_eval, pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_eval, pred, average="weighted", zero_division=0),
            "MCC": matthews_corrcoef(y_eval, pred)
        }, pred

    if selected == "Compare all models":
        rows = []
        prediction_map = {}
        for name, model in models.items():
            row, pred = evaluate(name, model)
            rows.append(row)
            prediction_map[name] = pred
        table = pd.DataFrame(rows).set_index("Model")
        st.subheader("All-model comparison")
        st.dataframe(table.style.format("{:.4f}"), use_container_width=True)
        chosen_for_detail = st.selectbox("Show detailed report for", list(models.keys()))
        pred = prediction_map[chosen_for_detail]
        current_model = models[chosen_for_detail]
    else:
        row, pred = evaluate(selected, models[selected])
        table = pd.DataFrame([row]).set_index("Model")
        st.subheader(f"Metrics — {selected}")
        st.dataframe(table.style.format("{:.4f}"), use_container_width=True)
        chosen_for_detail = selected
        current_model = models[selected]

    st.subheader(f"Confusion matrix — {chosen_for_detail}")
    labels = np.arange(len(CLASS_NAMES))
    cm = confusion_matrix(y_eval, pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    st.dataframe(cm_df, use_container_width=True)

    st.subheader(f"Classification report — {chosen_for_detail}")
    report = classification_report(y_eval, pred, labels=labels, target_names=CLASS_NAMES,
                                  zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
else:
    st.info("Upload a labelled CSV containing the feature columns and target column to begin.")
    st.caption("Use the project's test_data.csv as a sample input.")
