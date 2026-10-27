
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Dry Bean Analytics | Model Benchmark Studio",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# 2. Colouring Schema
# --------------------------------------------------
st.markdown("""
<style>
    /* Main application background */
    .stApp {
        background-color: #FFFFFF;
        color: #142D4E;
    }

    /* Main headings */
    h1, h2, h3 {
        color: #142D4E !important;
    }

    /* Introductory text */
    .stApp p {
        color: #34495E;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #EAF2F8;
    }

    /* Buttons */
    .stButton > button {
        background-color: #087F8C;
        color: #FFFFFF;
        border: 1px solid #087F8C;
        border-radius: 8px;
    }

    .stButton > button:hover {
        background-color: #066570;
        color: #FFFFFF;
        border-color: #066570;
    }

    /* CSV uploader */
    [data-testid="stFileUploader"] {
        background-color: #EAF2F8;
        border: 1px solid #B7D4E8;
        border-radius: 10px;
        padding: 10px;
    }

    /* Dropdown controls */
    [data-baseweb="select"] > div {
        border-color: #087F8C;
        border-radius: 8px;
    }

    /* Information and warning messages */
    [data-testid="stAlert"] {
        border-radius: 8px;
    }

    /* Tables and dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #D4E2EE;
        border-radius: 8px;
    }

    /* Captions */
    [data-testid="stCaptionContainer"] {
        color: #526B82;
    }

    /* Subtle separator */
    hr {
        border-color: #D4E2EE;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. Application heading and introduction
# --------------------------------------------------
st.markdown("""
<h1 style="
    color: #142D4E;
    font-size: 2.35rem;
    font-weight: 750;
    margin-bottom: 0.2rem;
">
    🫘 Dry Bean Analytics
    <span style="color: #087F8C;">
        | ML Model Benchmark Studio
    </span>
</h1>
""", unsafe_allow_html=True)

st.markdown(
    "Explore and benchmark five machine-learning classifiers "
    "using labelled Dry Bean data, with performance metrics "
    "and detailed classification reports."
)

st.divider()

# --------------------------------------------------
# 4. Load project metadata and saved models
# --------------------------------------------------
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"

with open(ROOT / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

MODEL_FILES = metadata["model_filenames"]
FEATURES = metadata["feature_columns"]
TARGET = metadata["target_column"]
CLASS_NAMES = metadata["class_names"]

LABEL_TO_INT = {
    label: i for i, label in enumerate(CLASS_NAMES)
}


@st.cache_resource
def load_models():
    """Load saved models without retraining."""
    return {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }


models = load_models()

# --------------------------------------------------
# 5. Upload labelled test data
# --------------------------------------------------
st.subheader("📂 Upload Test Dataset")

st.write(
    "Upload a labelled CSV file to evaluate the saved classifiers. "
    "The models are loaded from disk; no retraining occurs."
)

uploaded = st.file_uploader(
    "Upload a labelled test CSV",
    type=["csv"]
)

if uploaded is not None:

    # Read the uploaded CSV file
    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Unable to read the uploaded CSV: {exc}")
        st.stop()

    # Check that all required features and target are present
    missing = [
        column
        for column in FEATURES + [TARGET]
        if column not in data.columns
    ]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Check that the uploaded file contains records
    if data.empty:
        st.error("The uploaded CSV contains no rows.")
        st.stop()

    # Check for missing values in required columns
    if data[FEATURES + [TARGET]].isnull().any().any():
        st.error(
            "The uploaded CSV contains missing values "
            "in the required features or target column."
        )
        st.stop()

    # Validate target labels
    unknown_labels = sorted(
        set(data[TARGET].astype(str)) - set(CLASS_NAMES)
    )

    if unknown_labels:
        st.error(
            "Target column contains unknown class labels: "
            f"{unknown_labels}"
        )
        st.stop()

    # Prepare evaluation features and target labels
    X_eval = data[FEATURES].copy()

    y_eval = (
        data[TARGET]
        .astype(str)
        .map(LABEL_TO_INT)
        .to_numpy()
    )

    # --------------------------------------------------
    # 6. Model selection
    # --------------------------------------------------
    st.subheader("⚙️ Select Classification Model")

    selected = st.selectbox(
        "Select a classifier",
        ["Compare all models"] + list(models.keys())
    )

    # --------------------------------------------------
    # 7. AUC calculation
    # --------------------------------------------------
    def auc_score(model, X, y):
        """Calculate ROC AUC where supported."""
        try:
            scores = model.predict_proba(X)

            if len(CLASS_NAMES) == 2:
                return roc_auc_score(y, scores[:, 1])

            return roc_auc_score(
                y,
                scores,
                multi_class="ovr",
                average="macro",
                labels=np.arange(len(CLASS_NAMES))
            )

        except (
            ValueError,
            TypeError,
            AttributeError
        ):
            return float("nan")

    # --------------------------------------------------
    # 8. Model evaluation
    # --------------------------------------------------
    def evaluate(name, model):
        """Calculate the six assignment evaluation metrics."""
        pred = model.predict(X_eval)

        result = {
            "Model": name,
            "Accuracy": accuracy_score(y_eval, pred),
            "AUC": auc_score(model, X_eval, y_eval),
            "Precision": precision_score(
                y_eval,
                pred,
                average="weighted",
                zero_division=0
            ),
            "Recall": recall_score(
                y_eval,
                pred,
                average="weighted",
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_eval,
                pred,
                average="weighted",
                zero_division=0
            ),
            "MCC": matthews_corrcoef(y_eval, pred)
        }

        return result, pred

    # --------------------------------------------------
    # 9. Compare all models or show one model
    # --------------------------------------------------
    if selected == "Compare all models":

        rows = []
        prediction_map = {}

        for name, model in models.items():
            row, pred = evaluate(name, model)
            rows.append(row)
            prediction_map[name] = pred

        table = pd.DataFrame(rows).set_index("Model")

        st.subheader("📊 All-Model Performance Comparison")

        st.dataframe(
            table.style.format("{:.4f}"),
            use_container_width=True
        )

        chosen_for_detail = st.selectbox(
            "Show detailed report for",
            list(models.keys())
        )

        pred = prediction_map[chosen_for_detail]
        current_model = models[chosen_for_detail]

    else:

        row, pred = evaluate(selected, models[selected])

        table = pd.DataFrame([row]).set_index("Model")

        st.subheader(f"📈 Performance Metrics — {selected}")

        st.dataframe(
            table.style.format("{:.4f}"),
            use_container_width=True
        )

        chosen_for_detail = selected
        current_model = models[selected]

    # --------------------------------------------------
    # 10. Confusion matrix
    # --------------------------------------------------
    st.divider()

    st.subheader(
        f"🔎 Confusion Matrix — {chosen_for_detail}"
    )

    labels = np.arange(len(CLASS_NAMES))

    cm = confusion_matrix(
        y_eval,
        pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    # --------------------------------------------------
    # 11. Classification report
    # --------------------------------------------------
    st.subheader(
        f"📋 Classification Report — {chosen_for_detail}"
    )

    report = classification_report(
        y_eval,
        pred,
        labels=labels,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }),
        use_container_width=True
    )

    # --------------------------------------------------
    # 12. Completion message
    # --------------------------------------------------
    st.success(
        "Evaluation completed successfully. "
        "All results are based on the uploaded labelled test data."
    )

else:

    # Message displayed before a file is uploaded
    st.info(
        "Upload a labelled CSV containing the required feature "
        "columns and target column to begin evaluation."
    )

    st.caption(
        "You can use the project's test_data.csv as a sample input."
    )

# --------------------------------------------------
# 13. Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Dry Bean Analytics | Machine Learning Assignment 2 "
    "— Five-classifier evaluation and comparison"
)
