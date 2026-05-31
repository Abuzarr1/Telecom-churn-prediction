# Customer Churn Analysis & Prediction — Final Project Report

This is the finalized, comprehensive report of the **Customer Churn Prediction System**. The system successfully executes an end-to-end data science and machine learning product suite—from PostgreSQL database ingestion, through feature engineering and tree-based model training, to explainable AI and a live, premium interactive dashboard.

---

## 🎯 What We Built

An end-to-end machine learning system that identifies which customers are highly likely to cancel their subscription—**before they actually do**—packaged inside a premium, visual web interface.

### The Churn Command Center (Streamlit Dashboard)

Our frontend interface connects directly to PostgreSQL and run live inferences with our trained XGBoost model:

*   **Executive Dashboard Page**: Displays active KPIs (Total Evaluated Clients, Overall Churn Rate, MRR at Risk), Category distributions (Bar chart of 🟢 Low, 🟡 Medium, 🔴 High cohorts), and a searchable client grid.
*   **Customer Lookup Page**: Select any customer account to check their contract plan, monthly charges, and check their exact predictive features (Failed payments, Days late, Tenure).
*   **What-If Simulator Page**: Slide sliders representing customer engagement behaviors (e.g. increase billing failures or switch contract types) and see how the model updates its churn predictions in real-time.

---

## 🗺️ The Project Roadmap (100% Completed)

Every phase of the project has been fully developed, validated, and implemented:

```mermaid
gantt
    title Customer Churn Project Phases - Completed
    dateFormat  YYYY-MM-DD
    section Complete
    Step 1 - Environment Setup   :done,    des1, 2026-05-25, 2026-05-26
    Step 2 - Database Creation   :done,    des2, 2026-05-26, 2026-05-26
    Step 3 - Data Loading        :done,    des3, 2026-05-26, 2026-05-27
    Step 4 - Feature Engineering :done,    des4, 2026-05-27, 2026-05-27
    Step 5 - XGBoost Training    :done,    des5, 2026-05-27, 2026-05-27
    Step 6 - Evaluation & Saving  :done,    des6, 2026-05-27, 2026-05-27
    Streamlit Dashboard UI       :done,    des7, 2026-05-27, 2026-05-30
```

### ✅ Step 1 — Set up the Environment
Configured a stable Python runtime (`Python 3.11` inside `venv`) with psycopg2-binary, pandas, sqlalchemy, scikit-learn, xgboost, shap, and streamlit fully installed.

### ✅ Step 2 — Created the Database
Designed and initialized PostgreSQL `churn_db` with 5 relational tables (`customers`, `subscriptions`, `payments`, `usage_events`, and `support_tickets`).

### ✅ Step 3 — Loaded the Data
Wrote an idempotent ETL pipeline `load_data.py` that cleanly flushes prior records (avoiding unique violations) and seeds **7,043 real customers** and their historical invoicing logs.

### ✅ Step 4 — Feature Engineering
Wrote **[features.py](file:///Users/macbookpro/Desktop/Customer_Churn/src/features.py)** to query PostgreSQL and compile predictive features (Tenure, missed payments, contract contract types) into `data/features.csv` and the `customer_features` table.

### ✅ Step 5 — Train XGBoost
Wrote **[train.py](file:///Users/macbookpro/Desktop/Customer_Churn/src/train.py)** to train an `XGBClassifier` (saving weights to `models/churn_model.joblib`) and export global explainability charts to `outputs/shap_summary.png` using **SHAP**.

### ✅ Step 6 — Evaluate & Save
Wrote **[predict.py](file:///Users/macbookpro/Desktop/Customer_Churn/src/predict.py)** to score all 7,043 accounts and save risk levels to database table `churn_predictions` and `outputs/churn_predictions.csv`.

### ✅ Streamlit Dashboard Web UI
Wrote **[app.py](file:///Users/macbookpro/Desktop/Customer_Churn/src/app.py)** and launched it on `http://localhost:8501`, connecting live data science outcomes to a visual CRM app.

---

## 🎨 SHAP Global Explainability Chart

The model utilizes **SHAP (SHapley Additive exPlanations)** to ensure calculations are explainable. The chart shows that late payments and billing failures are the strongest predictors of client churn:

![XGBoost SHAP Summary Plot](/Users/macbookpro/.gemini/antigravity/brain/cfa0ffa7-b5e0-40e7-9c16-3915ab74fd83/shap_summary.png)

---

## 🛠️ Complete Project Directory Structure

```text
Customer_Churn/
├── requirements.txt         # Package dependencies
├── sql/
│   └── schema.sql           # Initial database schema
├── src/
│   ├── load_data.py         # Step 3: Raw Kaggle CSV ➔ PostgreSQL Tables
│   ├── features.py          # Step 4: SQL Aggregate Joins ➔ customer_features table & CSV
│   ├── train.py             # Step 5: Model training, metrics, & SHAP png
│   ├── predict.py           # Step 6: Risk inference ➔ churn_predictions table & CSV
│   └── app.py               # Streamlit interactive Churn Command Center Web UI
├── data/
│   ├── telco_churn.csv      # Raw CSV customer data
│   └── features.csv         # Computed feature matrix
├── models/
│   └── churn_model.joblib   # Serialized trained XGBoost model
└── outputs/
    ├── shap_summary.png     # Visual SHAP explainability plot
    └── churn_predictions.csv # Exported scored predictions matrix
```

---

## 🖥️ Live Dashboard & Database Integration

### 1. Launch the Streamlit Dashboard
To run the interactive web interface locally, execute the following command:
```bash
venv/bin/streamlit run src/app.py
```
Open **`http://localhost:8501`** in your browser to access the Executive KPIs, Search Profiler, and What-If Simulator!

### 2. Business Integration Queries (PostgreSQL)
Because predictions are committed directly to PostgreSQL, marketing teams can easily query high-risk segments:
```sql
-- Count customers by risk segment
SELECT risk_level, COUNT(*) AS count, ROUND(AVG(churn_probability)*100, 2) || '%' AS avg_probability
FROM churn_predictions
GROUP BY risk_level;
```
