import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

def main():
    print("🚀 Starting Step 4 — Synchronized Feature Engineering...")

    # Load environment variables and connect
    load_dotenv()
    DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(DB_URL)

    # 1. Load SQL Query from sql/feature_query.sql to keep features in sync
    sql_path = "sql/feature_query.sql"
    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"🚨 SQL query file not found at {sql_path}.")
    
    print(f"📄 Reading features query from {sql_path}...")
    with open(sql_path, 'r') as f:
        feature_query = f.read()

    print("📊 Querying database and extracting aggregate signals...")
    with engine.connect() as conn:
        df = pd.read_sql(text(feature_query), conn)
    
    print(f"✅ Extracted {len(df)} customer records.")

    # Rename 'churned' target column to 'label' as expected by our training pipeline
    df = df.rename(columns={'churned': 'label'})

    # 2. Categorical Preprocessing and Mapping for plan_type
    print("🧠 Encoding features for XGBoost...")
    
    # Map plan types to ordinal integers (basic -> 1, pro -> 2, enterprise -> 3)
    plan_mapping = {'basic': 1, 'pro': 2, 'enterprise': 3}
    df['plan_type_code'] = df['plan_type'].map(plan_mapping).fillna(1)
    
    # One-hot encode plan_type for maximum flexibility in tree models
    df = pd.get_dummies(df, columns=['plan_type'], prefix='plan', dtype=int)

    # Define clean column order
    cols = [
        'customer_id', 'label', 'tenure_days', 'avg_monthly_fee', 'num_subscriptions',
        'failed_payments', 'payment_failure_pct', 'avg_days_late', 'refunded_payments', 'total_payments',
        'plan_type_code', 'plan_basic', 'plan_pro', 'plan_enterprise'
    ]
    
    # Ensure all columns exist, if some one-hot columns are missing (e.g. if a category wasn't present)
    for col in cols:
        if col not in df.columns:
            df[col] = 0
            
    df_features = df[cols].copy()

    # 3. Save feature matrix back to database (table: customer_features)
    print("💾 Saving feature dataset back to PostgreSQL (table: customer_features)...")
    df_features.to_sql('customer_features', engine, if_exists='replace', index=False)

    # 4. Save feature matrix to CSV for local ML training access
    csv_path = "data/features.csv"
    print(f"📁 Saving feature dataset to local CSV: {csv_path}...")
    df_features.to_csv(csv_path, index=False)

    # 5. Display brief summary of features
    print("\n--- 📈 Feature Engineering Summary ---")
    print(f"Total Rows: {len(df_features)}")
    print(f"Columns: {list(df_features.columns)}")
    print("\nClass Balance (Churn):")
    churn_counts = df_features['label'].value_counts()
    for label, count in churn_counts.items():
        pct = (count / len(df_features)) * 100
        status = "Churned" if label == 1 else "Retained"
        print(f"  {status}: {count} ({pct:.2f}%)")
    
    print("\nFeature Averages:")
    print(f"  Average Tenure (Days): {df_features['tenure_days'].mean():.1f}")
    print(f"  Average Monthly Fee: ${df_features['avg_monthly_fee'].mean():.2f}")
    print(f"  Average Failed Payments: {df_features['failed_payments'].mean():.2f}")
    print(f"  Average Days Late: {df_features['avg_days_late'].mean():.2f}")
    print("--------------------------------------")
    print("\n🎉 Step 4 Complete!")

if __name__ == '__main__':
    main()
