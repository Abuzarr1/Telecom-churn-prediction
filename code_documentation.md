# Technical Architecture & Code Specification Document

This document provides a highly detailed, professional, file-by-file code specification of the **Telecom Customer Churn Prediction System**. It serves as an engineering manual to help you explain the database design, data pipeline mechanics, machine learning mathematics, and interface deployment layers in interviews or technical reviews.

---

## 🏗️ System Architecture Overview

The system is structured as an end-to-end, reproducible data science product:

```text
Customer_Churn/
├── requirements.txt         # Project dependency definitions
├── .env                     # Database secrets & environment variables
├── sql/
│   ├── schema.sql           # Database schema definition (DDL)
│   └── feature_query.sql    # Analytical feature extraction queries (CTEs)
├── src/
│   ├── load_data.py         # Step 1: Idempotent ETL Data Loader
│   ├── features.py          # Step 2: Feature Engineering & Preprocessing Pipeline
│   ├── train.py             # Step 3: Model Training, Cross-Validation, & Evaluation
│   └── cli_simulator.py     # Step 4: Interactive Python CLI Simulator
├── models/
│   └── churn_model.pkl      # Serialized XGBoost model weights (Joblib)
└── outputs/
    ├── model_eval.png       # Model ROC curve evaluation chart
    └── shap_importance.png  # Global SHAP feature importance summary chart
```

---

## 📂 File-by-File Technical Specifications

---

### 1. Project Dependencies & Credentials

#### 📄 [requirements.txt](file:///Users/macbookpro/Desktop/Customer_Churn/requirements.txt)
*   **Purpose**: Pinpoints the exact libraries required to execute the pipeline, ensuring clean reproducibility across environments.
*   **Key Dependencies**:
    *   `psycopg2-binary`: High-performance PostgreSQL adapter for Python.
    *   `sqlalchemy`: Database abstraction layer (SQL Toolkit and ORM).
    *   `pandas` & `numpy`: Core data structures for matrix manipulation and aggregations.
    *   `scikit-learn`: Split functions, diagnostic scoring metrics, and cross-validation tools.
    *   `xgboost`: Extreme Gradient Boosting classifier library.
    *   `shap`: Game-theoretic model explainability (SHapley Additive exPlanations).
    *   `joblib`: Object serialization library for loading and saving models.
    *   `python-dotenv`: Automated loading of `.env` configurations.

#### 📄 [.env](file:///Users/macbookpro/Desktop/Customer_Churn/.env)
*   **Purpose**: Decouples database credentials from code files, adhering to security best practices (12-Factor App design).
*   **Variables**:
    *   `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`: Direct connection settings for PostgreSQL.

---

### 2. Database & Query Layer (`sql/`)

#### 📄 [sql/schema.sql](file:///Users/macbookpro/Desktop/Customer_Churn/sql/schema.sql)
*   **Purpose**: Defines the normalized, relational SQL tables and constraints required to house raw telecom operations data.
*   **Schema Details**:
    *   `customers`: Primary master containing core accounts, signup dates, and binary churn results.
    *   `subscriptions`: Connects to customers via foreign key constraints, containing plan name details and recurring fees.
    *   `payments`: Tracks invoicing histories, status flags (`paid`, `failed`, `refunded`), and payment delinquency (`days_late`).
    *   `usage_events` & `support_tickets`: Capture customer interaction logs (session active time, support severity) to enable future behavioral signal scaling.
*   **Key Indexes**: Installs secondary B-Tree indexes on lookup keys (e.g. `customer_id`, `sub_id`, `status`) to ensure feature extraction queries perform efficiently over large datasets.

#### 📄 [sql/feature_query.sql](file:///Users/macbookpro/Desktop/Customer_Churn/sql/feature_query.sql)
*   **Purpose**: High-performance aggregate query that performs relational joins to extract customer billing, compliance, and tenure behaviors.
*   **Code Mechanics**:
    *   Uses **Common Table Expressions (CTEs)** to isolate calculations:
        *   `sub_features`: Group-by aggregations calculating average recurring fees, historical packages, and active tenure days.
        *   `payment_features`: Computes payment frequencies, counts failures (`status = 'failed'`), tracks refunded accounts, calculates the late payment average (`avg_days_late`), and computes the dynamic failure percentage:
            $$\text{payment\_failure\_pct} = 100 \times \frac{\text{failed\_payments}}{\text{total\_payments}}$$
    *   The outer query left-joins the `customers` base table with these CTE matrices, returning a flat table mapping each customer ID to their corresponding target label (`churned::INT`) and numerical variables.

---

### 3. Ingestion & Feature Extraction Layer (`src/`)

#### 📄 [src/load_data.py](file:///Users/macbookpro/Desktop/Customer_Churn/src/load_data.py)
*   **Purpose**: Serves as the Extract-Transform-Load (ETL) ingestion engine. It imports the raw Kaggle dataset, performs initial column cleaning, and populates the PostgreSQL database.
*   **Code Mechanics**:
    *   **Idempotency Fix**: Connects to the database and runs a cascading truncation:
        ```sql
        TRUNCATE TABLE payments, subscriptions, customers CASCADE;
        ```
        This flushes all tables, making the script completely re-runnable without violating `UNIQUE` email or primary key constraints.
    *   Reads `data/telco_churn.csv` into a Pandas DataFrame.
    *   Cleans column headers (stripping spaces, mapping to lowercase) and formats missing data (coercing spaces inside `totalcharges` to `0.0`).
    *   Splits records and uses `to_sql(..., if_exists='append')` to seed normalized rows into the `customers`, `subscriptions`, and `payments` tables sequentially.

#### 📄 [src/features.py](file:///Users/macbookpro/Desktop/Customer_Churn/src/features.py)
*   **Purpose**: Orchestrates the analytical feature construction pipeline, serving as the bridge between raw SQL tables and machine learning inputs.
*   **Code Mechanics**:
    *   Loads `sql/feature_query.sql` directly from disk and executes the query against PostgreSQL using a SQLAlchemy connection.
    *   Renames the target database column `churned` to `label`.
    *   **Ordinal Encoding**: Maps categories inside `plan_type` (`basic`, `pro`, `enterprise`) to ordered scale values `[1, 2, 3]`.
    *   **One-Hot Encoding**: Performs `pd.get_dummies` on `plan_type` to construct separate columns `plan_basic`, `plan_pro`, and `plan_enterprise` (crucial for tree-based models to understand contract categories).
    *   Saves the resulting feature matrix back to PostgreSQL (table `customer_features`) and exports a flat matrix file to `data/features.csv` for training access.

---

### 4. Machine Learning & Prediction Layer (`src/`)

#### 📄 [src/train.py](file:///Users/macbookpro/Desktop/Customer_Churn/src/train.py)
*   **Purpose**: The central machine learning pipeline. It loads features, balances classes, fits an extreme gradient boosting model with early stopping, runs CV routines, evaluates diagnostic metrics, and logs visual charts.
*   **Key Algorithms & Mathematical Constructs**:
    *   **Class Imbalance Adjuster**: Computes `scale_pos_weight` dynamically using active class sizes:
        ```python
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
        ```
        This tells XGBoost to scale the positive class gradients, preventing the model from ignoring churn events due to minority class bias.
    *   **Early Stopping Resolution**: Sets `early_stopping_rounds=10` directly inside the `XGBClassifier` constructor (modern XGBoost 1.6+ API). This monitors performance on a testing split and automatically halts training when validation loss stops improving, completely preventing overfitting.
    *   **Rigorous Cross-Validation**: Instantiates a separate model `cv_model` without early stopping, and executes a **Stratified 5-Fold Cross-Validation** via `cross_val_score(..., scoring='roc_auc')` to calculate unbiased out-of-fold metrics (averaging 0.9592 ROC-AUC).
    *   **Diagnostic Plotting**: Calculates Receiver Operating Characteristic (ROC) coordinates (FPR and TPR) via Scikit-Learn and plots the curve, exporting it to **`outputs/model_eval.png`**.
    *   **Explainable AI**: Fits a SHAP `TreeExplainer` on the testing cohort, generating a feature importance plot that is saved to **`outputs/shap_importance.png`**.
    *   **Model Weights**: Serializes model weights to **`models/churn_model.pkl`**.
    *   **Live Prediction Sync**: Scores the entire database and writes individual customer IDs, scores, and risk classifications (`risk_level`: 🔴 High, 🟡 Medium, 🟢 Low) to PostgreSQL table **`churn_predictions`**.

#### 📄 [src/predict.py](file:///Users/macbookpro/Desktop/Customer_Churn/src/predict.py)
*   **Purpose**: Independent prediction and scoring pipeline.
*   **Code Mechanics**:
    *   Loads the serialized classifier `models/churn_model.pkl`.
    *   Queries `customer_features` from the database.
    *   Runs model inference (`predict_proba`) across all records, maps risk thresholds (🔴 High $\ge 70\%$, 🟡 Medium $\ge 30\%$, 🟢 Low otherwise).
    *   Persists the results to PostgreSQL table `churn_predictions` and outputs a flat audit file to `outputs/churn_predictions.csv` for CRM integrations.

---

### 5. Product Deployment Layer (`src/`)

#### 📄 [src/cli_simulator.py](file:///Users/macbookpro/Desktop/Customer_Churn/src/cli_simulator.py)
*   **Purpose**: The interactive command-line interface. It provides an executive console, customer lookups, and real-time what-if scenario simulations in pure Python.
*   **Code Mechanics**:
    *   **Main Menu Loop**: A standard Python `while True` loop that displays console choices and reads input prompts.
    *   **Executive Dashboard Menu**: Connects to PostgreSQL using SQLAlchemy, queries customer metrics, calculates active churn rates and MRR at risk, and prints the top 10 highest risk outreach targets.
    *   **Customer Lookup Menu**: Accepts a Customer ID or name, queries the relational database, and prints key metrics like tenure days and failed payments alongside their calculated model risk score.
    *   **What-If Churn Simulator Menu**:
        *   Prompts the user to input details such as customer tenure, monthly charge, payment failures, plan type, and average days late.
        *   Constructs a single-row Pandas DataFrame matching the exact features, formats, and columns order required by the trained XGBoost model:
            ```python
            input_df = pd.DataFrame([{ ... }])
            input_df = input_df[cols_order]
            ```
        *   Feeds the structured input dataframe directly into `model.predict_proba(input_df)[0, 1]` to perform live inference, printing the resulting risk percentage and color-coded tier feedback.
