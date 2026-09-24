# Diabetes Risk Predictor — Streamlit Deployment Bundle

This folder is a self-contained deployment package: an `app.py` script plus
the trained model artifacts (pickle files) it loads. No other files from the
analysis notebook are required to run it.

## Contents
- `app.py` — the Streamlit app
- `logreg_model.pkl` — trained LogisticRegression model
- `scaler.pkl` — fitted StandardScaler (feature scaling)
- `imputer.pkl` — fitted SimpleImputer (median imputation for missing values)
- `feature_names.pkl` — ordered list of the 8 input feature names
- `metrics.csv` — test-set performance metrics, shown in the app's "About" panel
- `requirements.txt` — Python package dependencies

## Option A: Run locally

1. Open a terminal in this folder.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```
   streamlit run app.py
   ```
4. Streamlit will open the app in your browser (usually at `http://localhost:8501`).

## Option B: Deploy online (Streamlit Community Cloud) — free, gives you a shareable link

1. Create a GitHub repository (e.g. `diabetes-risk-predictor`) and push everything
   in this folder to it (`app.py`, all four `.pkl` files, `requirements.txt`).
   ```
   git init
   git add .
   git commit -m "Initial deployment"
   git branch -M main
   git remote add origin https://github.com/<your-username>/diabetes-risk-predictor.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, select your repository, branch (`main`), and set the
   main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` and launch
   the app automatically, giving you a public URL like:
   `https://<your-app-name>.streamlit.app`
5. Share that link — anyone can open it in a browser, no installation needed.

That URL is what you'd submit as the "cloud link" for deployment.
