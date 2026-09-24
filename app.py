"""
Streamlit app: Diabetes risk prediction using a trained Logistic Regression model.
Run locally with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# ------------------------------------------------------------------
# Load trained artifacts (produced by analysis.py)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("logreg_model.pkl")
    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, imputer, feature_names

model, scaler, imputer, feature_names = load_artifacts()

st.title("🩺 Diabetes Risk Predictor")
st.write(
    "This app uses a **Logistic Regression** model trained on the Pima Indians "
    "Diabetes dataset to estimate the probability that a patient has diabetes, "
    "based on diagnostic measurements."
)

st.divider()
st.subheader("Enter patient measurements")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin (mu U/mL)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

st.caption(
    "Tip: enter 0 for a measurement if it wasn't recorded — the app will impute it "
    "using the median value learned from the training data, matching how the model was trained."
)

if st.button("Predict", type="primary", use_container_width=True):
    raw = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, dpf, age]], columns=feature_names)

    # Replicate training-time preprocessing: 0 in these columns == missing
    zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    raw[zero_as_missing] = raw[zero_as_missing].replace(0, np.nan)

    imputed = pd.DataFrame(imputer.transform(raw), columns=feature_names)
    scaled = pd.DataFrame(scaler.transform(imputed), columns=feature_names)

    proba = model.predict_proba(scaled)[0, 1]
    pred = model.predict(scaled)[0]

    st.divider()
    st.subheader("Result")

    if pred == 1:
        st.error(f"⚠️ **Higher risk of diabetes** — estimated probability: {proba:.1%}")
    else:
        st.success(f"✅ **Lower risk of diabetes** — estimated probability: {proba:.1%}")

    st.progress(min(max(proba, 0.0), 1.0))
    st.caption(
        "This is a statistical estimate from a model trained on a small historical "
        "dataset, not a medical diagnosis. Please consult a healthcare professional."
    )

st.divider()
with st.expander("About this model"):
    try:
        metrics = pd.read_csv("metrics.csv", index_col=0).squeeze("columns")
        st.write("**Test-set performance:**")
        st.write(
            f"- Accuracy: {metrics['accuracy']:.3f}\n"
            f"- Precision: {metrics['precision']:.3f}\n"
            f"- Recall: {metrics['recall']:.3f}\n"
            f"- F1-score: {metrics['f1']:.3f}\n"
            f"- ROC-AUC: {metrics['roc_auc']:.3f}"
        )
    except FileNotFoundError:
        st.write("Run `analysis.py` first to generate model metrics.")
    st.write(
        "Trained with scikit-learn `LogisticRegression` on standardized, "
        "median-imputed features from the Pima Indians Diabetes dataset."
    )
