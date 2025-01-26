import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

st.title("Clustering Analysis Tool")
st.markdown("""
    **Features:**
    - Load and preprocess your dataset
    - Standardize the data for better clustering performance
    - Visualize clustering results using various algorithms:
        - K-Means
        - Agglomerative Clustering
        - Spectral Clustering
        - DBSCAN
        - Gaussian Mixture Model (GMM)
        - Birch Clustering
    - Evaluate clustering performance using metrics such as Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index.
    - Download clustered data.
    - Compare multiple clustering algorithms.

    **Instructions:**
    1. Upload your dataset (CSV format).
    2. Adjust clustering parameters as needed.
    3. Select the desired clustering algorithm from the sidebar to visualize.
    4. Explore the visualizations and metrics to gain insights from your data.
""")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("First few rows of the dataset:")
    st.write(data.head())
else:
    st.stop()

data_numeric = data.select_dtypes(include=[float, int])
if data_numeric.empty:
    st.write("No numeric data found. Please upload a dataset with numeric columns.")
    st.stop()
if data_numeric.isnull().sum().sum() > 0:
    st.write("The dataset contains missing values. Please handle them before proceeding.")
else:
    st.write("No missing values detected.")

st.write("### Data Description")
st.write(data_numeric.describe())

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

sample_size = min(500, len(data_scaled))
data_sample = data_scaled[:sample_size]

st.sidebar.header("Clustering Algorithm Configuration")
algorithm = st.sidebar.selectbox("Select Clustering Algorithm", 
                                  ['K-Means', 'Agglomerative', 'Spectral', 'DBSCAN', 'GMM', 'Birch'])

if algorithm == 'K-Means':
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_sample)

elif algorithm == 'Agglomerative':
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = agglo.fit_predict(data_sample)

elif algorithm == 'Spectral':
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    clusters = spectral.fit_predict(data_sample)

elif algorithm == 'DBSCAN':
    eps = st.sidebar.slider("Epsilon (eps)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    min_samples = st.sidebar.slider("Minimum Samples (min_samples)", min_value=1, max_value=10, value=5)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data_sample)

elif algorithm == 'GMM':
    n_components = st.sidebar.slider("Number of Components", min_value=1, max_value=10, value=3)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    clusters = gmm.fit_predict(data_sample)

elif algorithm == 'Birch':
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
    birch = Birch(n_clusters=n_clusters)
    clusters = birch.fit_predict(data_sample)

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_sample)

clustered_sample_data = pd.DataFrame(data_sample, columns=data_numeric.columns)
clustered_sample_data['PCA1'] = data_pca[:, 0]
clustered_sample_data['PCA2'] = data_pca[:, 1]
clustered_sample_data['Cluster'] = clusters

st.write("### Clustering Results Visualization")
plt.figure(figsize=(10, 6))
plt.scatter(clustered_sample_data['PCA1'], clustered_sample_data['PCA2'], c=clustered_sample_data['Cluster'], cmap='viridis', edgecolor='k')
plt.title(f'PCA Visualization with {algorithm} Clustering')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.colorbar(label='Cluster')
st.pyplot(plt)

if algorithm in ['K-Means', 'Agglomerative', 'Spectral', 'GMM', 'Birch']:
    st.write("### Clustering Performance Metrics")
    st.write(f"Silhouette Score: {silhouette_score(data_sample, clusters)}")
    st.write(f"Calinski-Harabasz Index: {calinski_harabasz_score(data_sample, clusters)}")
    st.write(f"Davies-Bouldin Index: {davies_bouldin_score(data_sample, clusters)}")

if st.button("Download Clustered Data"):
    clustered_data_csv = clustered_sample_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=clustered_data_csv, file_name='clustered_data.csv', mime='text/csv')

# Adding Feature Importance Analysis Section
st.sidebar.header("Feature Importance Configuration")
importance_method = st.sidebar.selectbox("Select Importance Algorithm", 
                                         ['None', 'RandomForest', 'XGBoost'])

if importance_method != 'None':
    if importance_method == 'RandomForest':
        st.write("### Feature Importance with RandomForest")
        rf_clf = RandomForestClassifier(random_state=42)
        rf_clf.fit(data_sample, clusters)
        importances = rf_clf.feature_importances_
        
    elif importance_method == 'XGBoost':
        st.write("### Feature Importance with XGBoost")
        xgb_clf = xgb.XGBClassifier(random_state=42)
        xgb_clf.fit(data_sample, clusters)
        importances = xgb_clf.feature_importances_
        
    # Displaying Feature Importances
    importance_df = pd.DataFrame({'Feature': data_numeric.columns, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    st.write(importance_df)

    # Visualizing feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title(f'Feature Importances ({importance_method})')
    st.pyplot(plt)

if st.sidebar.checkbox("Show Help"):
    st.sidebar.write("### Help")
    st.sidebar.write("1. Upload your dataset: The application accepts CSV files with numeric data.")
    st.sidebar.write("2. Select a clustering algorithm and adjust parameters using the sidebar.")
    st.sidebar.write("3. View the results and performance metrics.")
    st.sidebar.write("4. Download the clustered data for further analysis.")

