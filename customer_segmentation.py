import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import plotly.express as px

# Load dataset
df = pd.read_csv("retail_sales_dataset.csv")

# Cleaning
df = df.drop_duplicates().dropna()

# Encode categorical
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Feature Engineering
if 'AnnualIncome' in df.columns and 'Age' in df.columns:
    df['Income_per_Age'] = df['AnnualIncome'] / df['Age'].clip(lower=1)

if 'SpendingScore' in df.columns and 'AnnualIncome' in df.columns:
    df['Spend_to_Income_Ratio'] = df['SpendingScore'] / (df['AnnualIncome'].abs() + 1)

# Select features
X = df.drop(columns=['CustomerID'], errors='ignore')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
df['KMeans_Segment'] = kmeans.fit_predict(X_scaled)

agg = AgglomerativeClustering(n_clusters=4)
df['Agg_Segment'] = agg.fit_predict(X_scaled)

dbscan = DBSCAN(eps=1.5, min_samples=5)
df['DBSCAN_Segment'] = dbscan.fit_predict(X_scaled)

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = X_pca[:,0], X_pca[:,1]

fig = px.scatter(
    df, x="PCA1", y="PCA2",
    color="KMeans_Segment",
    hover_data=df.columns,
    title="Customer Segmentation Dashboard (KMeans)"
)
fig.show()

# Save output
df.to_csv("segmented_retail_sales.csv", index=False)
print("Segmented dataset saved as segmented_retail_sales.csv")
