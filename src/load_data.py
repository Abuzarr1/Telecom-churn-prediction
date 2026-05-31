import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)

# Clear existing data to prevent unique constraint violations on re-runs
print("🧹 Cleaning existing data in database...")
with engine.connect() as conn:
    conn.execute(text("TRUNCATE TABLE payments, subscriptions, customers CASCADE;"))
    conn.commit()

# Load CSV
df = pd.read_csv("data/telco_churn.csv")
print(f"Loaded {len(df)} rows")

# Clean column names
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

# Fix TotalCharges (has spaces instead of numbers for new customers)
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce').fillna(0)

# -------------------------------------------------------
# Map to customers table
# -------------------------------------------------------
customers = pd.DataFrame({
    'name':        df['customerid'],
    'email':       df['customerid'] + '@telco.com',
    'country':     'USA',
    'plan_type':   df['contract'].map({
                       'Month-to-month': 'basic',
                       'One year':       'pro',
                       'Two year':       'enterprise'
                   }),
    'signup_date': pd.Timestamp('2020-01-01') + pd.to_timedelta(df['tenure'] * 30, unit='D'),
    'churned':     df['churn'].map({'Yes': True, 'No': False}),
    'churn_date':  None
})

customers.to_sql('customers', engine, if_exists='append', index=False)
print(f"Inserted {len(customers)} customers")

# -------------------------------------------------------
# Map to subscriptions table
# (FIXED: subscription status is based on realistic signals,
#  NOT directly mapped from the churn label)
# -------------------------------------------------------
with engine.connect() as conn:
    cust_ids = pd.read_sql("SELECT customer_id, email FROM customers", conn)

df['email'] = df['customerid'] + '@telco.com'
df = df.merge(cust_ids, on='email', how='left')

np.random.seed(42)
churned_mask = df['churn'] == 'Yes'

# Realistic subscription status:
#   - Churned customers: 75% cancelled, 25% still marked active (recent churners)
#   - Retained customers: 5% cancelled (old plan swaps), 95% active
sub_status = pd.Series('active', index=df.index)
sub_status[churned_mask] = np.where(
    np.random.rand(churned_mask.sum()) < 0.75, 'cancelled', 'active'
)
sub_status[~churned_mask] = np.where(
    np.random.rand((~churned_mask).sum()) < 0.05, 'cancelled', 'active'
)

subscriptions = pd.DataFrame({
    'customer_id': df['customer_id'],
    'plan_name':   df['contract'],
    'monthly_fee': df['monthlycharges'],
    'start_date':  pd.Timestamp('2020-01-01'),
    'end_date':    None,
    'status':      sub_status
})

subscriptions.to_sql('subscriptions', engine, if_exists='append', index=False)
print(f"Inserted {len(subscriptions)} subscriptions")

# -------------------------------------------------------
# Map to payments table
# (FIXED: payment status and days_late use realistic
#  probabilistic distributions, NOT direct churn mapping)
# -------------------------------------------------------
with engine.connect() as conn:
    sub_ids = pd.read_sql("SELECT sub_id, customer_id FROM subscriptions", conn)

df = df.merge(sub_ids, on='customer_id', how='left')

# Realistic payment status:
#   - Churned customers: 30% failed, 5% refunded, 65% paid
#   - Retained customers: 3% failed, 2% refunded, 95% paid
pay_rand = np.random.rand(len(df))
pay_status = pd.Series('paid', index=df.index)

pay_status[churned_mask & (pay_rand < 0.30)] = 'failed'
pay_status[churned_mask & (pay_rand >= 0.30) & (pay_rand < 0.35)] = 'refunded'

pay_status[~churned_mask & (pay_rand < 0.03)] = 'failed'
pay_status[~churned_mask & (pay_rand >= 0.03) & (pay_rand < 0.05)] = 'refunded'

# Realistic days_late:
#   - Churned customers: mean=8, std=5 (clipped 0-30)
#   - Retained customers: mean=1, std=2 (clipped 0-10)
days_late = pd.Series(0, index=df.index, dtype=float)
days_late[churned_mask] = np.clip(
    np.random.normal(8, 5, churned_mask.sum()), 0, 30
).round(0)
days_late[~churned_mask] = np.clip(
    np.random.normal(1, 2, (~churned_mask).sum()), 0, 10
).round(0)

payments = pd.DataFrame({
    'sub_id':       df['sub_id'],
    'payment_date': pd.Timestamp('2024-01-01'),
    'amount':       df['monthlycharges'],
    'status':       pay_status,
    'days_late':    days_late.astype(int)
})

payments.to_sql('payments', engine, if_exists='append', index=False)
print(f"Inserted {len(payments)} payments")

print("\n✅ All data loaded successfully (with realistic synthetic signals)!")
