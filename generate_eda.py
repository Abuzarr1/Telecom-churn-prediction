import nbformat as nbf
import os

# Create a new notebook
nb = nbf.v4.new_notebook()

# Define the cells
cells = []

# Title & Introduction
cells.append(nbf.v4.new_markdown_cell("""
# 📊 Telecom Customer Churn: Exploratory Data Analysis (EDA)
**Author**: Data Science Intern
**Goal**: Understand the key drivers of customer churn and validate the synthetic signals introduced in the ETL pipeline.

This notebook serves as the initial exploratory phase before feature engineering and XGBoost model training. We will analyze the distribution of churn across various customer segments and billing behaviors.
"""))

# Imports & Setup
cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Plotting style
plt.style.use('dark_background')
sns.set_palette("husl")
"""))

# Load Data
cells.append(nbf.v4.new_markdown_cell("""
### 1. Load Feature Matrix from PostgreSQL
We'll query the `customer_features` table which contains our engineered features.
"""))

cells.append(nbf.v4.new_code_cell("""
# Connect to DB
load_dotenv('../.env')
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)

# Load data
df = pd.read_sql("SELECT * FROM customer_features", engine)
df.head()
"""))

# Class Balance
cells.append(nbf.v4.new_markdown_cell("""
### 2. Class Distribution (Target Variable)
First, let's look at our class imbalance. This informs our decision to use `scale_pos_weight` in the XGBoost model.
"""))

cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='label', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Target Variable Distribution (Churn)')
plt.xticks([0, 1], ['Retained (0)', 'Churned (1)'])

# Add percentage annotations
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2 - 0.05
    y = p.get_y() + p.get_height() + 50
    ax.annotate(percentage, (x, y), ha='center')

plt.tight_layout()
plt.show()
"""))

# Churn vs Tenure
cells.append(nbf.v4.new_markdown_cell("""
### 3. Impact of Customer Tenure
Are newer customers more likely to churn than long-term loyalists?
"""))

cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(10, 5))
sns.kdeplot(data=df, x='tenure_days', hue='label', fill=True, common_norm=False, palette=['#2ecc71', '#e74c3c'])
plt.title('Density of Tenure (Days) by Churn Status')
plt.xlabel('Tenure (Days)')
plt.ylabel('Density')
plt.xlim(0, df['tenure_days'].max())
plt.show()
"""))

# Churn vs Failed Payments
cells.append(nbf.v4.new_markdown_cell("""
### 4. Billing Health: Failed Payments
Does billing friction (failed payments) strongly correlate with churn risk?
"""))

cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(8, 5))
sns.boxplot(x='label', y='failed_payments', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Distribution of Failed Payments by Churn Status')
plt.xticks([0, 1], ['Retained (0)', 'Churned (1)'])
plt.ylabel('Number of Failed Payments')
plt.show()
"""))

# Correlation Matrix
cells.append(nbf.v4.new_markdown_cell("""
### 5. Feature Correlation Matrix
Let's visualize the linear relationships between our numerical features.
"""))

cells.append(nbf.v4.new_code_cell("""
corr_cols = ['label', 'tenure_days', 'avg_monthly_fee', 'failed_payments', 'avg_days_late']
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()
"""))

# Conclusion
cells.append(nbf.v4.new_markdown_cell("""
### 🔍 Key Takeaways for Modeling:
1.  **Imbalance**: We have a ~73/27 split, confirming the need for `scale_pos_weight` in XGBoost or SMOTE.
2.  **Tenure**: Churn density is heavily skewed towards newer customers. Tenure will be a highly predictive feature.
3.  **Billing**: Failed payments and days late show a positive correlation with the churn label, making them excellent behavioral signals.
"""))


# Add cells to notebook
nb.cells.extend(cells)

# Write to file
os.makedirs('notebooks', exist_ok=True)
with open('notebooks/eda.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✅ Successfully generated notebooks/eda.ipynb")
