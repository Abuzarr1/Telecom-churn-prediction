import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

# Set page configuration for a premium visual feel
st.set_page_config(
    page_title="Churn Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# DATABASE & MODEL CACHING
# ---------------------------------------------------------

@st.cache_resource
def get_db_engine():
    load_dotenv()
    DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return create_engine(DB_URL)

@st.cache_resource
def load_churn_model():
    model_path = "models/churn_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_kpi_data():
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

@st.cache_data
def load_customer_features(cust_id):
    engine = get_db_engine()
    query = f"""
    SELECT cf.*, c.name AS kaggle_customer_id
    FROM customer_features cf
    JOIN customers c ON cf.customer_id = c.customer_id
    WHERE cf.customer_id = {cust_id};
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)

# ---------------------------------------------------------
# DASHBOARD LAYOUT & ENGINE RUNNING
# ---------------------------------------------------------

st.sidebar.title("🛡️ Churn Control")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation Menu",
    ["📊 Executive Dashboard", "👤 Customer Profile Lookup", "🎛️ What-If Simulator"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Targeted Retain Tip**:\n\nMonth-to-month contracts have historically generated our highest churn risks. Upgrading them to annual plans is our best mitigation tool."
)

model = load_churn_model()

if model is None:
    st.warning("⚠️ Churn model is not trained yet! Please run `venv/bin/python src/train.py` first.")
else:
    try:
        df_kpis = load_kpi_data()
        
        # ---------------------------------------------------------
        # PAGE 1 — EXECUTIVE COMMAND DASHBOARD
        # ---------------------------------------------------------
        if page == "📊 Executive Dashboard":
            st.title("📊 Executive Churn Command Center")
            st.markdown("Monitor high-level churn behaviors, revenue at risk, and segment distributions live from PostgreSQL.")
            st.markdown("---")

            # Metrics
            total_clients = len(df_kpis)
            high_risk_df = df_kpis[df_kpis['risk_level'] == '🔴 High']
            churn_rate = len(high_risk_df) / total_clients if total_clients > 0 else 0
            mrr_at_risk = high_risk_df['monthly_fee'].sum()

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Evaluated Clients", f"{total_clients:,}")
            with m2:
                st.metric("Model-Flagged Churn Rate", f"{churn_rate:.2%}")
            with m3:
                st.metric("Monthly Recurring Revenue (MRR) at Risk", f"${mrr_at_risk:,.2f}")

            st.markdown("---")

            # Charts and Tables layout
            c1, c2 = st.columns([1, 1.5])
            
            with c1:
                st.subheader("📦 Risk Category Distribution")
                risk_counts = df_kpis['risk_level'].value_counts()
                # Create a neat streamlit bar chart
                chart_data = pd.DataFrame({
                    'Volume': [risk_counts.get('🟢 Low', 0), risk_counts.get('🟡 Medium', 0), risk_counts.get('🔴 High', 0)]
                }, index=['🟢 Low Risk', '🟡 Medium Risk', '🔴 High Risk'])
                st.bar_chart(chart_data, height=300)
                
            with c2:
                st.subheader("🔍 Search Scored Customer Accounts")
                search_q = st.text_input("Search by Customer ID (Kaggle or Database ID):", "")
                
                # Filter dataframe
                filtered_df = df_kpis.copy()
                if search_q:
                    filtered_df = filtered_df[
                        (filtered_df['kaggle_customer_id'].str.contains(search_q, case=False, na=False)) |
                        (filtered_df['customer_id'].astype(str).str.contains(search_q))
                    ]
                
                # Display high-risk list or clean search table
                display_cols = {
                    'customer_id': 'DB ID',
                    'kaggle_customer_id': 'Kaggle ID',
                    'plan_name': 'Contract Plan',
                    'monthly_fee': 'Monthly Charge',
                    'churn_probability': 'Churn Score',
                    'risk_level': 'Risk Tier'
                }
                st.dataframe(
                    filtered_df[list(display_cols.keys())].rename(columns=display_cols).sort_values(by='Churn Score', ascending=False),
                    hide_index=True,
                    use_container_width=True,
                    height=280
                )

        # ---------------------------------------------------------
        # PAGE 2 — CUSTOMER PROFILE LOOKUP
        # ---------------------------------------------------------
        elif page == "👤 Customer Profile Lookup":
            st.title("👤 Customer Risk Profile Lookup")
            st.markdown("Enter or select a customer ID to retrieve their exact predictive features and risk score.")
            st.markdown("---")

            # Search bar with autocomplete list of Kaggle IDs
            search_list = df_kpis['kaggle_customer_id'].tolist()
            selected_kaggle_id = st.selectbox("Select Customer profile to examine:", search_list)

            if selected_kaggle_id:
                # Find DB customer_id corresponding to selected Kaggle ID
                cust_row = df_kpis[df_kpis['kaggle_customer_id'] == selected_kaggle_id].iloc[0]
                cust_id = cust_row['customer_id']
                
                # Load features from database
                feats = load_customer_features(cust_id).iloc[0]

                # Draw profile layout
                l1, l2 = st.columns([1, 1.2])

                with l1:
                    st.subheader(f"🆔 Profile Details: {selected_kaggle_id}")
                    st.write(f"**Database Account Key**: `CUST-{cust_id}`")
                    st.write(f"**Contract Plan**: `{cust_row['plan_name']}`")
                    st.write(f"**Active Status**: `🟢 Active`" if cust_row['risk_level'] != '🔴 High' else f"**Active Status**: `⚠️ At Churn Risk`")
                    st.write(f"**Monthly Billing Charge**: `${cust_row['monthly_fee']:.2f}`")
                    st.write(f"**Client Since**: `{cust_row['signup_date']}`")
                    
                    st.markdown("---")
                    
                    # Large colored metric for probability
                    prob = cust_row['churn_probability']
                    risk = cust_row['risk_level']
                    
                    st.markdown(f"### Predicted Churn Risk: `{prob:.2%}`")
                    if risk == '🔴 High':
                        st.error(f"**Tier: {risk}**\n\n**Loyalty Outreach Required!** Client exhibits critical indicators (e.g. month-to-month contracts and/or missed payments). Contact account manager immediately.")
                    elif risk == '🟡 Medium':
                        st.warning(f"**Tier: {risk}**\n\n**Warning Engagement Recommended.** Schedule call to discuss moving basic/medium packages to annual subscriptions.")
                    else:
                        st.success(f"**Tier: {risk}**\n\n**Healthy Account.** Client exhibits high engagement and billing compliance. Standard operations.")

                with l2:
                    st.subheader("📊 Engine Predictive Features (Postgres)")
                    
                    # Render nice table of the raw inputs that caused this classification
                    input_summary = pd.DataFrame({
                        'Feature Name': [
                            'Tenure Length (Days)',
                            'Monthly Charges ($)',
                            'Total Invoiced Payments',
                            'Failed Payments Count',
                            'Overdue Payments (Max Days Late)'
                        ],
                        'Account Value': [
                            f"{feats['tenure_days']:,} days",
                            f"${feats['monthly_fee']:.2f}",
                            f"{int(feats['total_payments_count'])} payments",
                            f"{int(feats['failed_payments_count'])} failures",
                            f"{int(feats['max_days_late'])} days overdue"
                        ]
                    })
                    st.table(input_summary)

        # ---------------------------------------------------------
        # PAGE 3 — WHAT-IF RETENTION SIMULATOR
        # ---------------------------------------------------------
        elif page == "🎛️ What-If Simulator":
            st.title("🎛️ What-If Retention Scenario Simulator")
            st.markdown("Modify account signals in real-time and observe how our trained XGBoost classifier updates its predictions instantly.")
            st.markdown("---")

            col_in, col_out = st.columns([1.2, 1])

            with col_in:
                st.subheader("⚙️ Adjust Customer Account Parameters")
                
                tenure_y = st.slider("Customer Tenure (Years):", 0.0, 6.0, 2.5, 0.1)
                tenure_days = int(tenure_y * 365)
                
                monthly_fee = st.slider("Monthly Charges ($):", 15.0, 125.0, 65.0, 1.0)
                
                total_payments = st.number_input("Total Payments Invoiced so far:", 0, 150, int(tenure_y * 12))
                
                failed_payments = st.slider("Number of Billing Failures (Failed Payments):", 0, 10, 0, 1)
                
                max_days_late = st.slider("Maximum Days Late on Payments:", 0, 30, 0, 1)
                
                plan_type = st.selectbox(
                    "Select Contract Plan Category:",
                    ["Month-to-month", "One year", "Two year"]
                )

            with col_out:
                st.subheader("🔮 Simulated Real-Time Prediction Outcome")
                st.write("Below is the output generated by our active XGBoost prediction pipeline:")
                
                # Map inputs to model features list
                plan_code = 1 if plan_type == "Month-to-month" else (2 if plan_type == "One year" else 3)
                p_basic = 1 if plan_type == "Month-to-month" else 0
                p_pro = 1 if plan_type == "One year" else 0
                p_ent = 1 if plan_type == "Two year" else 0

                # Form input frame (9 columns matching features.py schema)
                input_df = pd.DataFrame([{
                    'tenure_days': tenure_days,
                    'monthly_fee': monthly_fee,
                    'total_payments_count': total_payments,
                    'failed_payments_count': failed_payments,
                    'max_days_late': max_days_late,
                    'plan_type_code': plan_code,
                    'plan_basic': p_basic,
                    'plan_pro': p_pro,
                    'plan_enterprise': p_ent
                }])

                # Reorder columns explicitly to match model fit schema
                cols_order = ['tenure_days', 'monthly_fee', 'total_payments_count', 'failed_payments_count',
                              'max_days_late', 'plan_type_code', 'plan_basic', 'plan_pro', 'plan_enterprise']
                input_df = input_df[cols_order]

                # Run live inference
                pred_prob = model.predict_proba(input_df)[0, 1]
                
                # Render results nicely
                st.markdown("---")
                st.markdown(f"### Simulated Churn Score: `{pred_prob:.2%}`")
                
                # Risk level mapping
                if pred_prob >= 0.70:
                    st.error("🔴 **Classification: HIGH RISK COHORT**\n\nThis account exhibits severe signals correlated with cancellations. Recommend immediate plan price adjustment or contract upgrade options.")
                elif pred_prob >= 0.30:
                    st.warning("🟡 **Classification: MEDIUM RISK COHORT**\n\nAccount displays mild churn warning signs. Recommend email/SMS notifications with contract options.")
                else:
                    st.success("🟢 **Classification: LOW RISK COHORT**\n\nAccount is extremely stable and highly compliant.")
                
                st.markdown("---")
                st.markdown(
                    "💡 **Simulation Tip**: Try increasing the `Billing Failures` to `3` or switching the plan to `Month-to-month` to see how rapidly the risk score climbs!"
                )

    except Exception as e:
        st.error(f"🚨 Failed to load data from PostgreSQL. Error: {e}")
        st.info("💡 Please make sure PostgreSQL is running, and that you have loaded predictions using `venv/bin/python src/predict.py`.")
