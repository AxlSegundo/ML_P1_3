import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def select_features(df):
    selected_df = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    return selected_df

def normalize_data(df):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    return pd.DataFrame(normalized_data, columns=df.columns)

if __name__ == "__main__":
    file_path = "P1_3/Practica1_3/Mall_Customers.csv"
    output_path = "P1_3/Practica1_3/Selected_Mall_Customers.csv"
    
    df = pd.read_csv(file_path)
    selected_df = select_features(df)
    
    selected_df.to_csv(output_path, index=False)
    print(f"Datos normalizados guardados en {output_path}")
