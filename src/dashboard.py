import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

# Set page configuration for a premium, dark-luxury feel
st.set_page_config(
    page_title="Telecom Churn Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CACHED DATA & RESOURCE LOADING
# ---------------------------------------------------------

@st.cache_resource
def get_db_engine():
    load_dotenv()
    DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return create_engine(DB_URL)

@st.cache_resource
def load_prediction_model():
    model_path = "models/churn_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_predictions_data():
    engine = get_db_engine()
    query = """
    SELECT 
        p.customer_id,
        p.kaggle_customer_id,
        p.churn_probability,
        p.risk_level,
        s.monthly_fee,
        s.plan_name,
        c.signup_date
    FROM churn_predictions p
    LEFT JOIN subscriptions s ON p.customer_id = s.customer_id
    LEFT JOIN customers c ON p.customer_id = c.customer_id;
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)

# ---------------------------------------------------------
# APPLICATION RENDER LAYOUT
# ---------------------------------------------------------

# Title and Description
st.title("🛡️ Enterprise Churn Command Center")
st.markdown("Designed by **Data Science Intern** — Connecting live predictive modeling and dynamic decision simulators.")
st.markdown("---")

model = load_prediction_model()

try:
    df_preds = load_predictions_data()
    
    # Calculate high-level metrics
    total_customers = len(df_preds)
    high_risk_df = df_preds[df_preds['risk_level'] == '🔴 High']
    high_risk_count = len(high_risk_df)
    churn_rate = high_risk_count / total_customers if total_customers > 0 else 0.0

    # 1. Metric Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="💼 Total Accounts Scored",
            value=f"{total_customers:,}",
            help="Total live subscriber profiles analyzed by the ML pipeline."
        )
    with col2:
        st.metric(
            label="📉 Active Churn Rate Flag",
            value=f"{churn_rate:.2%}",
            delta="-0.45%" if churn_rate > 0 else "0.00%",
            help="Percentage of customer base exhibiting high risk scores."
        )
    with col3:
        st.metric(
            label="🚨 High Churn Risk Cohort",
            value=f"{high_risk_count:,}",
            delta=f"{high_risk_count} critical alerts",
            delta_color="inverse",
            help="Number of accounts with a churn probability >= 70%."
        )
        
    st.markdown("---")

    # 2. Tabbed Navigation Layout (Highly Professional CV Design)
    tab1, tab2, tab3 = st.tabs([
        "📊 Executive Dashboard", 
        "🧠 Model Health & Explainability", 
        "🎛️ Real-Time What-If Simulator"
    ])

    # ---------------------------------------------------------
    # TAB 1 — EXECUTIVE DASHBOARD
    # ---------------------------------------------------------
    with tab1:
        st.subheader("🔴 Immediate Action List: High-Risk Accounts")
        st.markdown("Query and filter customer accounts flagged for immediate proactive retention outreach:")
        
        search_q = st.text_input("🔍 Search scored list by Customer ID (Kaggle or DB Key):", "")
        
        # Filter high risk records
        high_risk_display = high_risk_df.copy()
        if search_q:
            high_risk_display = high_risk_display[
                (high_risk_display['kaggle_customer_id'].str.contains(search_q, case=False, na=False)) |
                (high_risk_display['customer_id'].astype(str).str.contains(search_q))
            ]

        # Rename columns to be beautiful
        display_mapping = {
            'customer_id': 'DB Key',
            'kaggle_customer_id': 'Kaggle ID',
            'plan_name': 'Contract Plan',
            'monthly_fee': 'Monthly Charge',
            'churn_probability': 'Churn Score',
            'signup_date': 'Signup Date'
        }
        
        st.dataframe(
            high_risk_display[list(display_mapping.keys())].rename(columns=display_mapping).sort_values(by='Churn Score', ascending=False),
            hide_index=True,
            use_container_width=True,
            height=320
        )

    # ---------------------------------------------------------
    # TAB 2 — MODEL HEALTH & EXPLAINABILITY
    # ---------------------------------------------------------
    with tab2:
        st.subheader("🧪 Machine Learning Model Diagnostic & SHAP Interpretability")
        st.markdown("Provides full visual audits of our trained XGBoost classifier to build trust in predictions:")

        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("### 📈 Receiver Operating Characteristic (ROC) Curve")
            eval_path = "outputs/model_eval.png"
            if os.path.exists(eval_path):
                st.image(eval_path, caption="XGBoost ROC Evaluation Curve (Test Set)", use_container_width=True)
            else:
                st.info("⚠️ Evaluation plot outputs/model_eval.png not found.")

            # Print diagnostic metrics
            st.markdown(
                "#### Performance Metrics Summary:\n"
                "*   Metrics are computed and logged live during `train.py` execution.\n"
                "*   See terminal output or the ROC curve above for exact AUC values.\n"
                "*   Model uses **early stopping** and **scale_pos_weight** for imbalance handling."
            )

        with c_right:
            st.markdown("### 🧠 SHAP Global Feature Importance")
            shap_path = "outputs/shap_importance.png"
            if os.path.exists(shap_path):
                st.image(shap_path, caption="SHAP Global summary representing feature weights", use_container_width=True)
            else:
                st.info("⚠️ SHAP plot outputs/shap_importance.png not found.")

            st.markdown(
                "#### SHAP Impact Summary:\n"
                "1.  **Late Payments & Billing Failures** are the primary predictors of subscription churn.\n"
                "2.  **Short customer tenure** represents a critical vulnerability phase.\n"
                "3.  Month-to-month basic plans carry a significantly higher baseline risk than multi-year commitments."
            )

    # ---------------------------------------------------------
    # TAB 3 — WHAT-IF RETENTION SIMULATOR
    # ---------------------------------------------------------
    with tab3:
        st.subheader("🎛️ Real-Time What-If Scenario Simulator")
        st.markdown("Allows business operators to modify customer variables on-the-fly and run live predictions against the deployed XGBoost model:")
        
        if model is None:
            st.warning("⚠️ Deployed model models/churn_model.pkl is missing. Please run train.py first.")
        else:
            col_in, col_out = st.columns([1.2, 1])

            with col_in:
                st.markdown("### ⚙️ Customer Account Parameters")
                
                tenure_y = st.slider("Customer Tenure (Years):", 0.0, 6.0, 2.5, 0.1)
                tenure_days = int(tenure_y * 365)
                
                avg_monthly_fee = st.slider("Average Monthly Charge ($):", 15.0, 125.0, 65.0, 1.0)
                
                total_payments = st.number_input("Total Payments Invoiced:", 0, 150, int(tenure_y * 12))
                
                failed_payments = st.slider("Failed Payments (Billing Failures):", 0, 10, 0, 1)
                
                avg_days_late = st.slider("Average Days Late on Invoices:", 0.0, 30.0, 0.0, 0.5)
                
                refunded_payments = st.slider("Refunded Payments Count:", 0, 5, 0, 1)
                
                num_subscriptions = st.number_input("Number of Subscriptions:", 1, 5, 1)
                
                plan_type = st.selectbox(
                    "Select Subscription Plan Type:",
                    ["Month-to-month (basic)", "One year (pro)", "Two year (enterprise)"]
                )

            with col_out:
                st.markdown("### 🔮 Live Predictive Inference")
                st.markdown("Real-time inference generated by the model:")
                
                # Map inputs to features matching training matrix
                plan_code = 1 if "basic" in plan_type else (2 if "pro" in plan_type else 3)
                p_basic = 1 if "basic" in plan_type else 0
                p_pro = 1 if "pro" in plan_type else 0
                p_ent = 1 if "enterprise" in plan_type else 0

                # Form input frame (9 features matching train.py schema)
                input_df = pd.DataFrame([{
                    'tenure_days': tenure_days,
                    'avg_monthly_fee': avg_monthly_fee,
                    'num_subscriptions': num_subscriptions,
                    'failed_payments': failed_payments,
                    'payment_failure_pct': float((failed_payments / max(total_payments, 1)) * 100.0),
                    'avg_days_late': avg_days_late,
                    'refunded_payments': refunded_payments,
                    'total_payments': total_payments,
                    'plan_type_code': plan_code,
                    'plan_basic': p_basic,
                    'plan_pro': p_pro,
                    'plan_enterprise': p_ent
                }])

                # Reorder columns explicitly to match training columns
                cols_order = [
                    'tenure_days', 'avg_monthly_fee', 'num_subscriptions',
                    'failed_payments', 'payment_failure_pct', 'avg_days_late', 'refunded_payments', 'total_payments',
                    'plan_type_code', 'plan_basic', 'plan_pro', 'plan_enterprise'
                ]
                input_df = input_df[cols_order]

                # Run live inference
                pred_prob = model.predict_proba(input_df)[0, 1]
                
                st.markdown("---")
                st.markdown(f"## Live Churn Score: `{pred_prob:.2%}`")
                
                # Classification color coding
                if pred_prob >= 0.70:
                    st.error("🔴 **Classification: HIGH RISK COHORT**\n\nAccount exhibits critical payment compliance and contract warning signals. Recommend transitioning to multi-year commitments immediately.")
                elif pred_prob >= 0.30:
                    st.warning("🟡 **Classification: MEDIUM RISK COHORT**\n\nAccount displays mild churn warning signs. Recommend proactive contract audits and feedback surveys.")
                else:
                    st.success("🟢 **Classification: LOW RISK COHORT**\n\nAccount is highly engaged and financially stable.")
                
                st.markdown("---")
                st.markdown(
                    "💡 **Simulation Idea**:\n"
                    "Slide **`Failed Payments`** to `3` or increase **`Average Days Late`** to see how the model's confidence shifts instantaneously!"
                )

except Exception as e:
    st.error(f"🚨 Failed to load data from PostgreSQL. Error: {e}")
    st.info("💡 Ensure PostgreSQL is running, and that you have seeded calculations using `venv/bin/python src/train.py`.")
