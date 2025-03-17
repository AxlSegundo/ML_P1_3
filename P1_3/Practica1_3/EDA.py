import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def explore_data(df):
    print("\nInformación del Dataset:")
    print(df.info())
    print("\nPrimeras filas del Dataset:")
    print(df.head())
    print("\nDescripción estadística:")
    print(df.describe())

    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    print("\nCurtosis y Asimetría:")
    for col in ['Annual Income (k$)', 'Spending Score (1-100)']:
        print(f"{col}:")
        print(f"  Curtosis: {kurtosis(df[col])}")
        print(f"  Asimetría: {skew(df[col])}")

def plot_distributions(df):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['Annual Income (k$)'], bins=20, kde=True, color='blue')
    plt.title('Distribución de Annual Income')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, color='green')
    plt.title('Distribución de Spending Score')
    
    plt.show()

def plot_boxplots(df):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(x=df['Annual Income (k$)'], color='green')
    plt.title('Diagrama de Caja - Ingreso Anual')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df['Spending Score (1-100)'], color='orange')
    plt.title('Diagrama de Caja - Puntuación de Gasto')
    
    plt.show()

def plot_scatter(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Gender'], palette='coolwarm')
    plt.title('Ingreso Anual vs Puntuación de Gasto')
    plt.xlabel('Ingreso Anual (k$)')
    plt.ylabel('Puntuación de Gasto (1-100)')
    plt.show()

if __name__ == "__main__":
    file_path = "P1_3/Practica1_3/Mall_Customers.csv"
    df = load_data(file_path)
    explore_data(df)
    plot_distributions(df)
    plot_boxplots(df)
    plot_scatter(df)