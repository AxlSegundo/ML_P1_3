import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # O ajusta al número de núcleos de tu CPU
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data(file_path):
    """Carga los datos normalizados desde un CSV."""
    return pd.read_csv(file_path)

def find_optimal_k(df, max_k=10):
    """Determina el número óptimo de clusters usando el método del codo y el índice de silueta."""
    distortions = []
    silhouette_scores = []
    k_values = range(2, max_k+1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df, labels))
    
    # Método del codo
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, distortions, marker='o', linestyle='--')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Distorsión (Inertia)')
    plt.title('Método del Codo')
    
    # Índice de silueta
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', color='red')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Índice de Silueta')
    plt.title('Índice de Silueta por Número de Clusters')
    
    plt.show()
    
    # Selecciona el k con el mayor índice de silueta
    optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    return optimal_k

def apply_kmeans(df, k):
    """Aplica el algoritmo K-Medias con el número óptimo de clusters."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df)
    return df, kmeans

def plot_clusters(df):
    """Visualiza los clusters resultantes en un gráfico de dispersión."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['Cluster'], palette='viridis', s=50)
    plt.xlabel('Annual Income (Normalizado)')
    plt.ylabel('Spending Score (Normalizado)')
    plt.title('Clusters de Clientes')
    plt.legend(title='Cluster')
    plt.show()

if __name__ == "__main__":
    file_path = "P1_3/Practica1_3/Selected_Mall_Customers.csv"
    df = load_data(file_path)
    
    optimal_k = find_optimal_k(df, max_k=10)  # Determinar número óptimo de clusters
    clustered_df, kmeans_model = apply_kmeans(df, optimal_k)
    
    clustered_df.to_csv("P1_3/Practica1_3/ClusteredS_Mall_Customers.csv", index=False)
    print("Clusters guardados en 'ClusteredS_Mall_Customers.csv'")
    
    plot_clusters(clustered_df)