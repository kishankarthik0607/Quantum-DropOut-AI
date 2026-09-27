# Quantum_DropOut_AI

Student retention analytics: explore the data, train a model, score a student, see why, and get suggested institutional interventions.

## Run

```bash
pip install -r requirements.txt
streamlit run main.py        # run from this folder so .streamlit/config.toml is picked up
```

Put your dataset at `data/student_dropout_data.csv`, or upload a CSV on the Overview page. The CSV needs a `Target` column with the values `Dropout`, `Enrolled` or `Graduate`. The original lookup paths (including your `C:/Users/swapn/...` ones) are still tried.

The first visit shows a login page. Create an account, then sign in.

## Structure

```
main.py                  entry point: auth gate -> top navigation -> page
styles/theme.css         the whole design system
static/Manrope.ttf       self-hosted font (SIL OFL, see OFL-Manrope.txt)
qdai/
  config.py              constants (TARGET_MAPPING, RF params, risk bands ... preserved from the original)
  data.py                loading, validation, preprocess_data (preserved), 80/20 split (preserved)
  ml.py                  train_model (preserved Random Forest), real metrics, comparison models
  explain.py             SHAP / LIME / permutation / partial dependence
  features.py            how each column appears in the prediction form
  interventions.py       rule-based interventions built from SHAP signals
  insights.py            dataset-derived findings (minimum group size 30)
  auth.py                local accounts (PBKDF2), independent of the ML code
  views/                 login, overview, explore, models, predict, insights
  ui/                    components and Plotly charts
legacy/main_original.py  your original single-file app, untouched
```

## What is real

Every number shown comes from the dataset, the trained model, or a stated rule.

- **Metrics** are computed on the 20% hold-out split (`random_state=42`). Precision, recall and F1 are weighted averages; ROC-AUC is macro one-vs-rest.
- **Risk score** is the model's predicted probability of `Dropout`. The Low / Medium / High bands (under 30%, 30-60%, 60%+) are fixed presentation thresholds, not calibrated cut-offs.
- **"Why this result"** is the SHAP contribution of each feature towards `Dropout`, in percentage points. Base rate plus contributions equals the predicted probability.
- **Interventions** appear only when a theme raises the dropout probability by at least 2 percentage points (or a direct fee/debt flag is present and risk is above Low). They are suggestions for an advisor, not clinical or psychological advice.
- **Comparison models** (Logistic Regression, Gradient Boosting) are trained on the same split. Random Forest is the only model used for predictions and explanations.

## Limitations

- Authentication is a local account store for a single deployment. A browser refresh signs you out, and the lockout is per browser session. For a public deployment use Streamlit's OIDC `st.login`.
- The design was tested on Streamlit 1.64. The CSS relies on Streamlit's `data-testid` attributes, which can change between versions.
- Course, application-mode and parental codes are shown as `Code N` unless the label is documented (marital status, and course codes when they match the public UCI scheme).
