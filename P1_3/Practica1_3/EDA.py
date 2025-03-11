import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Carga el dataset en un DataFrame de pandas."""
    df = pd.read_csv(file_path)
    return df

def explore_data(df):
    """Muestra información general sobre el dataset."""
    print("\nInformación del Dataset:")
    print(df.info())
    print("\nPrimeras filas del Dataset:")
    print(df.head())
    print("\nDescripción estadística:")
    print(df.describe())

def plot_distributions(df):
    """Grafica la distribución de Annual Income y Spending Score."""
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    sns.histplot(df['Annual Income (k$)'], bins=20, kde=True, color='blue')
    plt.title('Distribución de Annual Income')
    
    plt.subplot(1,2,2)
    sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, color='green')
    plt.title('Distribución de Spending Score')
    
    plt.show()

if __name__ == "__main__":
    file_path = "P1_3/Practica1_3/Mall_Customers.csv"
    df = load_data(file_path)
    explore_data(df)
    plot_distributions(df)
