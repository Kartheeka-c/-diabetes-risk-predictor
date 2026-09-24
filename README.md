# Diabetes Risk Predictor 🩺

A machine learning web app that predicts the likelihood of diabetes based on
diagnostic health measurements, built with **Logistic Regression** and
deployed with **Streamlit**.

This project was built as part of a Logistic Regression assignment covering
the full ML workflow: exploratory data analysis, preprocessing, model
building, evaluation, interpretation, and deployment.

## 🔗 Live App
👉 [Try it here](https://your-app-name.streamlit.app) <!-- replace with your actual Streamlit Cloud URL -->

## 📊 Dataset
[Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
— 768 patient records with 8 diagnostic features:

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/mL) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age in years |

Target: `Outcome` (1 = diabetes, 0 = no diabetes)

## 🧠 Model
- **Algorithm**: Logistic Regression (scikit-learn)
- **Preprocessing**: Disguised missing values (zeros in Glucose, BloodPressure,
  SkinThickness, Insulin, BMI) imputed with median values, then standardized
- **Train/test split**: 80/20, stratified by class

### Test Set Performance
| Metric | Score |
|---|---|
| Accuracy | 0.71 |
| Precision | 0.60 |
| Recall | 0.50 |
| F1-score | 0.55 |
| ROC-AUC | 0.81 |

### Key Predictors
Glucose and BMI are the strongest predictors of diabetes risk, followed by
Pregnancies, DiabetesPedigreeFunction, and Age — consistent with established
clinical risk factors.

## 🚀 Run Locally

```bash
git clone https://github.com/joshikareddy-07/diabetes-risk-predictor.git
cd diabetes-risk-predictor
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 📁 Repository Contents
```
├── app.py                  # Streamlit app
├── logreg_model.pkl        # Trained logistic regression model
├── scaler.pkl              # Fitted StandardScaler
├── imputer.pkl             # Fitted SimpleImputer (median strategy)
├── feature_names.pkl       # Ordered list of input features
├── requirements.txt        # Python dependencies
└── README.md
```

## ⚠️ Disclaimer
This tool provides a statistical estimate based on a small historical
dataset. It is **not a medical diagnosis**. Always consult a healthcare
professional for medical advice.
