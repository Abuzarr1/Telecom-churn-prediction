import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

def main():
    print("🚀 Starting Step 6 — Generating Customer Churn Predictions...")

    # Load environment variables and connect
    load_dotenv()
    DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(DB_URL)

    # 1. Load trained model (.pkl format)
    model_path = "models/churn_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"🚨 Model file not found at {model_path}. Please run train.py first.")
    
    print(f"🧠 Loading trained XGBoost model from {model_path}...")
    model = joblib.load(model_path)

    # 2. Query features and metadata (joining with customers to get the original Kaggle customerID/name)
    print("📊 Querying client feature sets and customer IDs...")
    query = """
    SELECT 
        c.name AS customer_name,
        cf.*
    FROM customer_features cf
    JOIN customers c ON cf.customer_id = c.customer_id;
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    if df.empty:
        raise ValueError("🚨 No customer features found in the database. Please run features.py first.")

    # 3. Calculate Predictions & Risk Levels
    print("🔮 Running churn predictions across all active accounts...")
    
    # Extract features corresponding to training schema
    # (Exclude non-feature columns: customer_name, customer_id, label)
    feature_cols = [col for col in df.columns if col not in ['customer_name', 'customer_id', 'label']]
    X = df[feature_cols]

    # Predict churn probability
    df['churn_probability'] = model.predict_proba(X)[:, 1]

    # Map probability to risk thresholds
    def get_risk_level(prob):
        if prob >= 0.70:
            return "🔴 High"
        elif prob >= 0.30:
            return "🟡 Medium"
        else:
            return "🟢 Low"

    df['risk_level'] = df['churn_probability'].apply(get_risk_level)

    # 4. Prepare target datasets for saving
    # Keep only target identifiers and prediction values
    predictions_df = df[['customer_id', 'customer_name', 'churn_probability', 'risk_level']].copy()
    
    # 5. Persist to PostgreSQL (table: churn_predictions)
    # We will rename customer_name to kaggle_customer_id for clarity
    predictions_df = predictions_df.rename(columns={'customer_name': 'kaggle_customer_id'})
    
    print("💾 Saving predictions back to PostgreSQL (table: churn_predictions)...")
    predictions_df.to_sql('churn_predictions', engine, if_exists='replace', index=False)
    
    # 6. Save a local CSV record
    output_csv = "outputs/churn_predictions.csv"
    predictions_df.to_csv(output_csv, index=False)
    print(f"📁 Exported prediction dataset to: {output_csv}")

    # 7. Print beautiful high-risk terminal lookup dashboard
    print("\n--- 🎯 High-Risk Customer Lookup Dashboard ---")
    print(f"{'Customer ID':<15} | {'Kaggle ID':<12} | {'Churn Prob':<12} | {'Risk Level':<12}")
    print("-" * 60)
    
    # Sort by probability descending to show highest risk first
    top_risk = predictions_df.sort_values(by='churn_probability', ascending=False).head(10)
    for _, row in top_risk.iterrows():
        print(f"CUST-{row['customer_id']:<10} | {row['kaggle_customer_id']:<12} | {row['churn_probability']:.4f}       | {row['risk_level']}")
    
    print("-" * 60)
    print(f"Total customers evaluated: {len(predictions_df)}")
    print(f"  🔴 High Risk count:   {len(predictions_df[predictions_df['risk_level'] == '🔴 High'])}")
    print(f"  🟡 Medium Risk count: {len(predictions_df[predictions_df['risk_level'] == '🟡 Medium'])}")
    print(f"  🟢 Low Risk count:    {len(predictions_df[predictions_df['risk_level'] == '🟢 Low'])}")
    print("----------------------------------------------")

    print("\n🎉 Step 6 Complete! The model predictions are now live and fully queryable.")

if __name__ == '__main__':
    main()
