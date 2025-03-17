import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Ajusta al número de núcleos de tu CPU
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def load_data(file_path):
    return pd.read_csv(file_path)

def find_optimal_k(df, max_k=10):
    distortions = []
    silhouette_scores = []
    davies_bouldin_scores = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df)
        

        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df, labels))
        davies_bouldin_scores.append(davies_bouldin_score(df, labels))
    

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(k_values, distortions, marker='o', linestyle='--')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Distorsión (Inertia)')
    plt.title('Método del Codo')

    plt.subplot(1, 3, 2)
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', color='red')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Índice de Silueta')
    plt.title('Índice de Silueta')
    

    plt.subplot(1, 3, 3)
    plt.plot(k_values, davies_bouldin_scores, marker='o', linestyle='--', color='green')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Coeficiente de Davies-Bouldin')
    plt.title('Coeficiente de Davies-Bouldin')
    
    plt.tight_layout()
    plt.show()
    
    optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    return optimal_k

def apply_kmeans(df, k):

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df)
    return df, kmeans

def plot_clusters(df):
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("husl", n_colors=len(df['Cluster'].unique()))
    sns.scatterplot(
        x=df.iloc[:, 0], 
        y=df.iloc[:, 1], 
        hue=df['Cluster'],  
        palette=palette,   
        s=100,             
        edgecolor='k',     
        legend='full'      
    )
    plt.xlabel('Annual Income (Normalizado)')
    plt.ylabel('Spending Score (Normalizado)')
    plt.title('Clusters de Clientes')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_clusters(df, kmeans_model):

    labels = kmeans_model.labels_
    silhouette_avg = silhouette_score(df, labels)
    davies_bouldin_avg = davies_bouldin_score(df, labels)
    
    print(f"\nEvaluación de Clusters:")
    print(f"  Índice de Silueta: {silhouette_avg:.3f}")
    print(f"  Coeficiente de Davies-Bouldin: {davies_bouldin_avg:.3f}")
    
    print("\nInterpretación:")
    print(f"  - Índice de Silueta: Un valor cercano a 1 indica clusters bien definidos.")
    print(f"  - Coeficiente de Davies-Bouldin: Un valor más cercano a 0 indica clusters mejor separados.")

if __name__ == "__main__":

    file_path = "P1_3/Practica1_3/Selected_Mall_Customers.csv"
    

    df = load_data(file_path)
    

    optimal_k = find_optimal_k(df, max_k=15)
    print(f"\nNúmero óptimo de clusters (k): {optimal_k}")
    
    clustered_df, kmeans_model = apply_kmeans(df, optimal_k)
    
    clustered_df.to_csv("P1_3/Practica1_3/ClusteredS_Mall_Customers.csv", index=False)
    print("Clusters guardados en 'ClusteredS_Mall_Customers.csv'")
    
    plot_clusters(clustered_df)
    
    evaluate_clusters(df, kmeans_model)
    cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
    print(cluster_counts)