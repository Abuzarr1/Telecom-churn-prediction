import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent GUI errors when saving plots
import matplotlib.pyplot as plt
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

def main():
    print("🚀 Starting Step 5 — Training XGBoost Model...")

    # Ensure output folders exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Load engineered features
    features_csv = "data/features.csv"
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"🚨 Features file not found at {features_csv}. Please run features.py first.")
    
    print(f"📁 Loading dataset from {features_csv}...")
    df = pd.read_csv(features_csv)
    
    # Separate customer_id, target, and features
    X = df.drop(columns=['customer_id', 'label'])
    y = df['label']

    print(f"🎯 Features dataset dimensions: {X.shape[0]} samples, {X.shape[1]} features.")
    
    # 2. Train-Test Split (80% Train, 20% Test)
    print("✂️ Splitting dataset into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Calculate dynamic scale_pos_weight to handle class imbalance
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️ Calculated scale_pos_weight: {scale_pos:.4f}")

    # 3. Initialize & Train XGBoost Classifier with early stopping and class weights
    print("🏋️ Training XGBoost Classifier with 10 early stopping rounds...")
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos,
        early_stopping_rounds=10  # Modern early stopping constructor parameter (fixes deprecation issue)
    )
    
    # Fit model with eval_set for early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    print("✅ Model fitting complete.")

    # 4. Run 5-Fold Cross Validation
    print("🔄 Running 5-Fold Stratified Cross-Validation...")
    # We clone the parameters to run CV, omitting early stopping rounds to prevent fits failing
    cv_model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"🏆 Average CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # 5. Model Evaluation
    print("📊 Evaluating model performance on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc_val = roc_auc_score(y_test, y_prob)

    print("\n--- 🏆 Model Performance Summary ---")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (Of predicted churns, how many actually churned)")
    print(f"  Recall:    {recall:.4f} (Of all actual churns, how many did we catch)")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc_val:.4f} (Core classification capacity)")
    print("-------------------------------------")
    
    # Print Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"  True Negatives (Retained caught): {cm[0][0]}")
    print(f"  False Positives (False Alarm):    {cm[0][1]}")
    print(f"  False Negatives (Missed Churn):   {cm[1][0]}")
    print(f"  True Positives (Churn caught):    {cm[1][1]}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    # 6. Save Evaluation ROC Plot to outputs/model_eval.png
    print("🎨 Generating model evaluation ROC curve plot...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_plot = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_plot:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('outputs/model_eval.png', dpi=300)
    plt.close()
    print("✅ Model evaluation plot outputs/model_eval.png generated successfully.")

    # 7. Serialize and Save Model Weights to models/churn_model.pkl (matching pkl name)
    model_path = "models/churn_model.pkl"
    print(f"💾 Saving trained model weights to {model_path}...")
    joblib.dump(model, model_path)
    print("✅ Model saved successfully.")

    # 8. Explainable AI with SHAP saved to outputs/shap_importance.png
    print("🧠 Computing SHAP explainability values...")
    try:
        # Create TreeExplainer (optimized for tree ensembles like XGBoost)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)

        # Plot and save global feature importance summary
        shap_plot_path = "outputs/shap_importance.png"
        print(f"🎨 Generating SHAP summary plot: {shap_plot_path}...")
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title("XGBoost Feature Importance (SHAP Global Summary)", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(shap_plot_path, dpi=300)
        plt.close()
        print("✅ SHAP plot outputs/shap_importance.png generated successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not generate SHAP plots. Error: {e}")

    # 9. Predictions Ingestion / Write-Back back to PostgreSQL
    print("💾 Running predictions across all active accounts & writing back to database...")
    try:
        load_dotenv()
        DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        engine = create_engine(DB_URL)

        # Compute predictions across all rows
        full_features = df[X.columns]
        df['churn_probability'] = model.predict_proba(full_features)[:, 1]

        # Map to risk levels
        def get_risk_level(prob):
            if prob >= 0.70: return "🔴 High"
            elif prob >= 0.30: return "🟡 Medium"
            return "🟢 Low"
        df['risk_level'] = df['churn_probability'].apply(get_risk_level)

        # Merge with customer names
        query_cust = "SELECT customer_id, name AS kaggle_customer_id FROM customers;"
        with engine.connect() as conn:
            df_cust = pd.read_sql(text(query_cust), conn)
        
        predictions_df = df[['customer_id', 'churn_probability', 'risk_level']].merge(df_cust, on='customer_id', how='left')
        predictions_df = predictions_df[['customer_id', 'kaggle_customer_id', 'churn_probability', 'risk_level']]

        predictions_df.to_sql('churn_predictions', engine, if_exists='replace', index=False)
        print("✅ Churn predictions successfully written back to PostgreSQL 'churn_predictions' table!")
    except Exception as e:
        print(f"⚠️ Warning: Prediction database write-back failed. Error: {e}")

    print("\n🎉 Step 5 Complete!")

if __name__ == '__main__':
    main()
