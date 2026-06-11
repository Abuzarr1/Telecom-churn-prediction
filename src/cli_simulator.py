import os
import sys
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Define terminal colors for professional visual styling
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_db_engine():
    load_dotenv()
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')
    
    if not all([db_user, db_password, db_host, db_port, db_name]):
        print(f"{Colors.RED}🚨 Missing database configuration in environment variables (.env file).{Colors.RESET}")
        return None
        
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    try:
        engine = create_engine(db_url)
        # Verify connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        print(f"{Colors.RED}🚨 Database connection failed: {e}{Colors.RESET}")
        return None

def load_model():
    model_path = "models/churn_model.pkl"
    if not os.path.exists(model_path):
        # Fallback to joblib model if pkl is not found
        model_path = "models/churn_model.joblib"
        
    if not os.path.exists(model_path):
        print(f"{Colors.YELLOW}⚠️  No trained model file found in 'models/'. Please run src/train.py first.{Colors.RESET}")
        return None
        
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"{Colors.RED}🚨 Failed to load ML model: {e}{Colors.RESET}")
        return None

def show_kpi_dashboard(engine):
    clear_screen()
    print(f"{Colors.HEADER}{Colors.BOLD}========================================================================{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}                   🛡️  EXECUTIVE CHURN COMMAND CENTER                     {Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}========================================================================{Colors.RESET}")
    
    if engine is None:
        print(f"{Colors.RED}Database engine is not connected. Cannot show dashboard.{Colors.RESET}")
        input(f"\nPress Enter to return to main menu...")
        return

    try:
        # Load prediction summary
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
            df = pd.read_sql(text(query), conn)

        if df.empty:
            print(f"{Colors.YELLOW}⚠️  No predictions found in the database. Run src/predict.py first.{Colors.RESET}")
            input(f"\nPress Enter to return to main menu...")
            return

        total_customers = len(df)
        high_risk_df = df[df['risk_level'].str.contains("High|🔴", case=False, na=False)]
        high_risk_count = len(high_risk_df)
        churn_rate = high_risk_count / total_customers if total_customers > 0 else 0.0
        mrr_at_risk = high_risk_df['monthly_fee'].sum()

        print(f"\n{Colors.BOLD}📊 Key Performance Indicators:{Colors.RESET}")
        print(f"  • {Colors.BOLD}Total Accounts Scored:{Colors.RESET}       {Colors.CYAN}{total_customers:,}{Colors.RESET}")
        print(f"  • {Colors.BOLD}Active Churn Rate Flag:{Colors.RESET}      {Colors.RED}{churn_rate:.2%}{Colors.RESET} ({high_risk_count} accounts)")
        print(f"  • {Colors.BOLD}Monthly Revenue (MRR) at Risk:{Colors.RESET} {Colors.YELLOW}${mrr_at_risk:,.2f}{Colors.RESET}")
        print("-" * 72)

        # Risk segments breakdown
        low_count = len(df[df['risk_level'].str.contains("Low|🟢", case=False, na=False)])
        med_count = len(df[df['risk_level'].str.contains("Medium|🟡", case=False, na=False)])
        
        print(f"{Colors.BOLD}📦 Risk Cohort Segments:{Colors.RESET}")
        print(f"  🟢 {Colors.GREEN}Low Risk:{Colors.RESET}    {low_count:<6} ({low_count/total_customers:.1%})")
        print(f"  🟡 {Colors.YELLOW}Medium Risk:{Colors.RESET} {med_count:<6} ({med_count/total_customers:.1%})")
        print(f"  🔴 {Colors.RED}High Risk:{Colors.RESET}   {high_risk_count:<6} ({high_risk_count/total_customers:.1%})")
        print("-" * 72)

        # Top 10 High Risk Accounts
        print(f"{Colors.RED}{Colors.BOLD}🔴 Top 10 Immediate Outreach Targets (High Risk):{Colors.RESET}")
        print(f"{'DB Key':<8} | {'Kaggle ID':<15} | {'Contract Plan':<15} | {'Monthly Fee':<12} | {'Churn Prob'}")
        print("-" * 72)
        top_10 = df.sort_values(by='churn_probability', ascending=False).head(10)
        for _, row in top_10.iterrows():
            plan = row['plan_name'] if row['plan_name'] else "Unknown"
            fee = f"${row['monthly_fee']:.2f}" if pd.notnull(row['monthly_fee']) else "$0.00"
            print(f"CUST-{row['customer_id']:<3} | {row['kaggle_customer_id']:<15} | {plan:<15} | {fee:<12} | {row['churn_probability']:.2%}")

    except Exception as e:
        print(f"{Colors.RED}🚨 Failed to read dashboard data: {e}{Colors.RESET}")
        
    input(f"\nPress Enter to return to main menu...")

def show_customer_lookup(engine):
    clear_screen()
    print(f"{Colors.HEADER}{Colors.BOLD}========================================================================{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}                 👤 CUSTOMER RISK PROFILE LOOKUP                         {Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}========================================================================{Colors.RESET}")
    
    if engine is None:
        print(f"{Colors.RED}Database engine is not connected. Cannot search customer profiles.{Colors.RESET}")
        input(f"\nPress Enter to return to main menu...")
        return

    search_id = input(f"\nEnter Customer ID (Kaggle Name or DB Key, e.g., 'CUST-10' or 'cust_10'): ").strip()
    if not search_id:
        return

    # Clean up standard formats (extract digit if it's CUST-12)
    db_id = None
    kaggle_id = None
    if search_id.lower().startswith('cust-') or search_id.lower().startswith('cust_'):
        try:
            db_id = int(search_id.split('-')[-1].split('_')[-1])
        except ValueError:
            pass
    else:
        # Try to parse it as integer first, else fallback to string lookup
        try:
            db_id = int(search_id)
        except ValueError:
            kaggle_id = search_id

    try:
        # Retrieve lookup lists
        if db_id is not None:
            query = f"""
            SELECT 
                p.customer_id, p.kaggle_customer_id, p.churn_probability, p.risk_level,
                s.monthly_fee, s.plan_name, c.signup_date
            FROM churn_predictions p
            LEFT JOIN subscriptions s ON p.customer_id = s.customer_id
            LEFT JOIN customers c ON p.customer_id = c.customer_id
            WHERE p.customer_id = {db_id};
            """
        else:
            query = f"""
            SELECT 
                p.customer_id, p.kaggle_customer_id, p.churn_probability, p.risk_level,
                s.monthly_fee, s.plan_name, c.signup_date
            FROM churn_predictions p
            LEFT JOIN subscriptions s ON p.customer_id = s.customer_id
            LEFT JOIN customers c ON p.customer_id = c.customer_id
            WHERE p.kaggle_customer_id ILIKE '{kaggle_id}';
            """
            
        with engine.connect() as conn:
            res = pd.read_sql(text(query), conn)

        if res.empty:
            print(f"\n{Colors.YELLOW}⚠️  No customer matching '{search_id}' was found in predictions database.{Colors.RESET}")
            input(f"\nPress Enter to retry/return...")
            return

        row = res.iloc[0]
        cust_id = row['customer_id']

        # Fetch detailed features
        query_feats = f"SELECT * FROM customer_features WHERE customer_id = {cust_id};"
        with engine.connect() as conn:
            feat_res = pd.read_sql(text(query_feats), conn)

        print(f"\n{Colors.BOLD}🆔 Profile Details: {Colors.CYAN}{row['kaggle_customer_id']}{Colors.RESET}")
        print("-" * 50)
        print(f"  • {Colors.BOLD}Database ID:{Colors.RESET}       CUST-{row['customer_id']}")
        print(f"  • {Colors.BOLD}Contract Plan:{Colors.RESET}     {row['plan_name'] or 'Unknown'}")
        print(f"  • {Colors.BOLD}Monthly Charge:{Colors.RESET}    ${row['monthly_fee']:.2f}")
        print(f"  • {Colors.BOLD}Signup Date:{Colors.RESET}       {row['signup_date']}")
        
        # Risk score output
        prob = row['churn_probability']
        risk_level = row['risk_level']
        
        if "High" in risk_level or "🔴" in risk_level:
            color = Colors.RED
            action = "⚠️  Loyalty outreach required immediately. High probability of contract termination."
        elif "Medium" in risk_level or "🟡" in risk_level:
            color = Colors.YELLOW
            action = "⚠️  Proactive loyalty follow-up recommended. Offer long-term contract upgrade options."
        else:
            color = Colors.GREEN
            action = "🟢  Account is healthy. Standard automated engagement."

        print("-" * 50)
        print(f"  • {Colors.BOLD}Predictive Churn Score:{Colors.RESET} {color}{Colors.BOLD}{prob:.2%}{Colors.RESET}")
        print(f"  • {Colors.BOLD}Assigned Risk Tier:{Colors.RESET}      {color}{Colors.BOLD}{risk_level}{Colors.RESET}")
        print(f"\n{Colors.BOLD}📢 Recommendation:{Colors.RESET} {action}")
        
        if not feat_res.empty:
            feats = feat_res.iloc[0]
            print(f"\n{Colors.BOLD}📊 Database Performance Signals (PostgreSQL):{Colors.RESET}")
            print(f"  - Tenure Length:         {int(feats['tenure_days']):,} days")
            print(f"  - Payment Failures:      {int(feats['failed_payments'])} failures")
            print(f"  - Overdue Payments Avg:  {feats['avg_days_late']:.1f} days late")
            print(f"  - Total Invoiced Count:  {int(feats['total_payments'])} invoices")

    except Exception as e:
        print(f"{Colors.RED}🚨 Profile retrieval failed: {e}{Colors.RESET}")

    input(f"\nPress Enter to return to menu...")

def show_whatif_simulator(model):
    clear_screen()
    print(f"{Colors.HEADER}{Colors.BOLD}========================================================================{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}                 🎛️  REAL-TIME WHAT-IF SIMULATOR                       {Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}========================================================================{Colors.RESET}")

    if model is None:
        print(f"{Colors.RED}Predictive model is not loaded. Cannot run what-if simulation.{Colors.RESET}")
        input(f"\nPress Enter to return to main menu...")
        return

    print(f"{Colors.CYAN}Simulate customer conditions on-the-fly and observe model predictions:{Colors.RESET}\n")

    # Helper function for safe float/int inputs with defaults
    def get_input(prompt, val_type=float, default=0.0, min_val=None, max_val=None):
        while True:
            raw = input(f"{prompt} [Default: {default}]: ").strip()
            if not raw:
                return default
            try:
                val = val_type(raw)
                if min_val is not None and val < min_val:
                    print(f"{Colors.YELLOW}⚠️  Must be >= {min_val}{Colors.RESET}")
                    continue
                if max_val is not None and val > max_val:
                    print(f"{Colors.YELLOW}⚠️  Must be <= {max_val}{Colors.RESET}")
                    continue
                return val
            except ValueError:
                print(f"{Colors.YELLOW}⚠️  Invalid input. Please enter a valid number.{Colors.RESET}")

    # Gather inputs
    tenure_y = get_input("1. Customer Tenure in Years (0.0 to 6.0)", float, 2.5, 0.0, 6.0)
    tenure_days = int(tenure_y * 365)
    
    avg_monthly_fee = get_input("2. Average Monthly Charge $ (15.0 to 125.0)", float, 65.0, 15.0, 125.0)
    
    total_payments = get_input("3. Total Payments Invoiced (0 to 150)", int, int(tenure_y * 12), 0, 150)
    
    failed_payments = get_input("4. Failed Payments / Billing Failures (0 to 10)", int, 0, 0, 10)
    
    avg_days_late = get_input("5. Average Days Late on Invoices (0.0 to 30.0)", float, 0.0, 0.0, 30.0)
    
    refunded_payments = get_input("6. Refunded Payments Count (0 to 5)", int, 0, 0, 5)
    
    num_subscriptions = get_input("7. Number of Subscriptions (1 to 5)", int, 1, 1, 5)

    print("\nSelect Plan Type Category:")
    print("  [1] Month-to-month (basic)")
    print("  [2] One year (pro)")
    print("  [3] Two year (enterprise)")
    plan_choice = get_input("Choose Option (1-3)", int, 1, 1, 3)

    # Map inputs to features matching training matrix
    plan_code = plan_choice
    p_basic = 1 if plan_choice == 1 else 0
    p_pro = 2 if plan_choice == 2 else 0
    p_ent = 3 if plan_choice == 3 else 0
    
    # Check features matching features.py
    # cols = ['tenure_days', 'avg_monthly_fee', 'num_subscriptions', 'failed_payments', 
    #         'payment_failure_pct', 'avg_days_late', 'refunded_payments', 'total_payments',
    #         'plan_type_code', 'plan_basic', 'plan_pro', 'plan_enterprise']
    
    payment_failure_pct = float((failed_payments / max(total_payments, 1)) * 100.0)

    # Form feature dataframe
    input_df = pd.DataFrame([{
        'tenure_days': tenure_days,
        'avg_monthly_fee': avg_monthly_fee,
        'num_subscriptions': num_subscriptions,
        'failed_payments': failed_payments,
        'payment_failure_pct': payment_failure_pct,
        'avg_days_late': avg_days_late,
        'refunded_payments': refunded_payments,
        'total_payments': total_payments,
        'plan_type_code': plan_code,
        'plan_basic': p_basic,
        'plan_pro': p_pro,
        'plan_enterprise': p_ent
    }])

    # Columns order matching XGBoost fit model
    cols_order = [
        'tenure_days', 'avg_monthly_fee', 'num_subscriptions',
        'failed_payments', 'payment_failure_pct', 'avg_days_late', 'refunded_payments', 'total_payments',
        'plan_type_code', 'plan_basic', 'plan_pro', 'plan_enterprise'
    ]
    input_df = input_df[cols_order]

    # Predict
    try:
        pred_prob = model.predict_proba(input_df)[0, 1]
        
        print("\n" + "=" * 50)
        print(f"🔮 {Colors.BOLD}LIVE INFERENCE OUTCOME:{Colors.RESET}")
        print(f"   Simulated Churn Score: {Colors.CYAN}{Colors.BOLD}{pred_prob:.2%}{Colors.RESET}")
        
        if pred_prob >= 0.70:
            print(f"   Risk Classification:   {Colors.RED}{Colors.BOLD}🔴 HIGH RISK COHORT{Colors.RESET}")
            print(f"   {Colors.RED}Model insight: Account exhibits high billing failure percentage and/or month-to-month contracts.{Colors.RESET}")
        elif pred_prob >= 0.30:
            print(f"   Risk Classification:   {Colors.YELLOW}{Colors.BOLD}🟡 MEDIUM RISK COHORT{Colors.RESET}")
            print(f"   {Colors.YELLOW}Model insight: Account shows intermediate risk signs. Proactive retention offer suggested.{Colors.RESET}")
        else:
            print(f"   Risk Classification:   {Colors.GREEN}{Colors.BOLD}🟢 LOW RISK COHORT{Colors.RESET}")
            print(f"   {Colors.GREEN}Model insight: Account is stable, compliant with billing, and displays high loyalty.{Colors.RESET}")
        print("=" * 50)
        
    except Exception as e:
        print(f"{Colors.RED}🚨 Simulated inference failed: {e}{Colors.RESET}")

    input(f"\nPress Enter to return to main menu...")

def main():
    # Load assets
    engine = load_db_engine()
    model = load_model()

    while True:
        clear_screen()
        print(f"{Colors.BLUE}{Colors.BOLD}========================================================================{Colors.RESET}")
        print(f"{Colors.BLUE}{Colors.BOLD}                 🛡️  TELECOM CHURN PREDICTION CLI SYSTEM                 {Colors.RESET}")
        print(f"{Colors.BLUE}{Colors.BOLD}========================================================================{Colors.RESET}")
        print(f"Active Model: {Colors.CYAN}XGBoost Classifier{Colors.RESET}")
        print(f"Active DB:    {Colors.CYAN}PostgreSQL (churn_db){Colors.RESET}")
        print("-" * 72)
        print("Please select an operation:")
        print(f"  {Colors.BOLD}[1]{Colors.RESET} View Executive Dashboard & High-Risk Accounts")
        print(f"  {Colors.BOLD}[2]{Colors.RESET} Search & Look Up Individual Customer Risk Profile")
        print(f"  {Colors.BOLD}[3]{Colors.RESET} Launch Live What-If Retention Scenario Simulator")
        print(f"  {Colors.BOLD}[4]{Colors.RESET} Exit System")
        print("-" * 72)
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            show_kpi_dashboard(engine)
        elif choice == '2':
            show_customer_lookup(engine)
        elif choice == '3':
            show_whatif_simulator(model)
        elif choice == '4':
            print(f"\n{Colors.GREEN}👋 Exiting Churn Command CLI. Thank you!{Colors.RESET}\n")
            sys.exit(0)
        else:
            print(f"{Colors.YELLOW}⚠️  Invalid option. Press Enter to try again...{Colors.RESET}")
            input()

if __name__ == '__main__':
    main()
