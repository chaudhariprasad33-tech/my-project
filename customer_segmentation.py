# %% [markdown]
# # End-to-End Data Science Project: Customer Segmentation
# This script is structured for VS Code IDE's Interactive Window or Jupyter Notebook.
# It performs end-to-end data analysis on the 'Mall Customer' dataset.

# %% [markdown]
# ## 1. Data Understanding
# Loading the libraries and understanding the basic shape and properties of the data.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Ignore minor warnings for cleaner outputs
warnings.filterwarnings('ignore')

# Load the dataset
# Ensure 'Mall Customer.csv' is in the same directory
try:
    df = pd.read_csv('Mall Customer.csv')
except FileNotFoundError:
    print("Please ensure 'Mall Customer.csv' is in the current directory.")

# Show dataset shape, columns, and basic info
print(f"Dataset Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

print("\nDataset Info:")
df.info()

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# %% [markdown]
# ## 2. Data Cleaning
# Handling missing values, duplicates, and converting categorical columns (Gender).

# %%
print("Missing values before cleaning:\n", df.isnull().sum())

# 1. Handle missing values
# Impute numerical missing values with the median to avoid outlier bias
numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Impute categorical missing values with Mode (if any)
if df['Gender'].isnull().sum() > 0:
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

# 2. Remove duplicates
duplicated_count = df.duplicated().sum()
if duplicated_count > 0:
    print(f"\nRemoving {duplicated_count} duplicate rows.")
    df.drop_duplicates(inplace=True)
else:
    print("\nNo duplicates found.")

# 3. Convert categorical variables into numerical format
# Mapping M -> 1 and F -> 0
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})

print("\nMissing values after cleaning:\n", df.isnull().sum())

# Drop CustomerID as it serves only as an identifier and has zero analytical value
if 'CustomerID' in df.columns:
    df.drop('CustomerID', axis=1, inplace=True)

# %% [markdown]
# ## 3. Data Processing & Feature Engineering
# Creating new meaningful features and scaling the inputs.

# %%
# 1. Create a new meaningful feature
# 'Income to Spending Ratio' = Annual Income / Spending Score
# Usefulness: This ratio highlights spending behavior. A high ratio equates to a cautious saver, 
# whereas a lower ratio indicates impulse or lavish spending relative to what they earn.
df['Income to Spending Ratio'] = df['Annual Income (k$)'] / (df['Spending Score (1-100)'] + 1e-5)

# 2. Normalize/Standardize numerical features
# Important for K-Means to ensure one variable (like Annual Income) 
# does not completely dominate another (like Age) due to magnitude differences.
scaler = StandardScaler()
features_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaled_features = scaler.fit_transform(df[features_to_scale])
df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale)

print("\nFeature Engineering completed. Preview of standardized data (Age, Income, Spending):")
print(df_scaled.head())

# %% [markdown]
# ## 4. Exploratory Data Analysis (EDA)
# Uncovering patterns using visualizations.

# %%
sns.set_theme(style="whitegrid")

# 4.1 Age vs Spending Score
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Age', y='Spending Score (1-100)', hue='Gender', data=df, palette='Set1')
plt.title('Age vs Spending Score (0=Female, 1=Male)')
plt.show()
# INSIGHT: Customers in the younger age bracket (20-35) regularly clock higher spending scores than seniors.

# 4.2 Annual Income vs Spending Score
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', data=df, palette='Set1')
plt.title('Annual Income vs Spending Score')
plt.show()
# INSIGHT: We can visually see clusters forming naturally in a 2D space. High-income earners are split into heavy spenders and stingy savers.

# 4.3 Distribution plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['Age'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Age Distribution')

sns.histplot(df['Annual Income (k$)'], kde=True, ax=axes[1], color='lightgreen')
axes[1].set_title('Annual Income Distribution')

sns.histplot(df['Spending Score (1-100)'], kde=True, ax=axes[2], color='salmon')
axes[2].set_title('Spending Score Distribution')
plt.show()
# INSIGHT: The customer age leans heavily toward millennials/GenZ. Income is heavily skewed to the right.

# 4.4 Correlation heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Income to Spending Ratio']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
# INSIGHT: Age and Spending Score share a notable negative correlation, confirming older clients tend to spend less.

# %% [markdown]
# ## 5. Customer Segmentation
# Grouping customers mathematically based on similar traits.

# %%
# We pass Age, Income, and Spending Score for K-Means clustering.
X_cluster = df_scaled[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method to determine optimal number of clusters
wcss = []
k_range = range(1, 11)
for i in k_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_cluster)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()
# INSIGHT: The line graph starts to flatten dramatically between 4 and 6. We will pick k=5 as the optimal point.

# Apply K-Means clustering
最优_kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = 最优_kmeans.fit_predict(X_cluster)

# Visualize clusters using Annual Income and Spending Score for 2D ease
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    hue='Cluster', 
    data=df, 
    palette='viridis', 
    s=100
)
plt.title('Customer Segments (K-Means Clustering)')
plt.legend(title='Cluster')
plt.show()

# %% [markdown]
# ## 6. Insights & Business Recommendations

# %%
"""
### Customer Segment Insights:
The clusters generally map to 5 unique personas:
- **Target/Champion Customers:** High Income, High Spending. These customers don't mind premium pricing and spend lavishly.
- **Careful/Saver Customers:** High Income, Low Spending. Have high purchasing authority but prefer frugality.
- **General Market:** Average Income, Average Spending. Forms the biggest majority of steady revenue.
- **Careless/Young Spenders:** Low Income, High Spending. They prefer engaging with retail despite limited capacity.
- **Sensible Shoppers:** Low Income, Low Spending. Highly budget-conscious and strictly need-based.

### 5 Actionable Business Recommendations:
1. **VIP Loyalty Programs (For Target Customers):** Reward your high-spending base with exclusive "early-bird" access to luxury arrivals to retain them.
2. **Quality & Value Marketing (For Careful Customers):** Pitch products emphasizing durability to break their savings habit using long-term value propositions.
3. **Flexible Payment & EMI (For Careless Spenders):** Support these low-income/high-affinity buyers with "Buy Now, Pay Later" (BNPL) schemes and bundles.
4. **Volume Discounts & Promos (For General Market):** Keep the mass traffic engaged on weekends using seasonal flash sales and BOGO offers.
5. **Clearance Alerts (For Sensible Shoppers):** Push SMS alerts for deep clearance and basic essential discounts directly to this group.
"""
print("\nSuccess! End-to-end data science project run dynamically complete.")
