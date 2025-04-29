import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_data(history_path, default_path):
    payment_history = pd.read_csv(history_path)
    payment_default = pd.read_csv(default_path)
    
    print(f"Payment history shape: {payment_history.shape}")
    print(f"Payment default shape: {payment_default.shape}")
    
    return payment_history, payment_default

def preprocess_default_data(default_df):
    df = default_df.copy()
    
    df['gender_encoded'] = df['gender'] - 1
    
    education_dummies = pd.get_dummies(df['education'], prefix='education')
    marital_dummies = pd.get_dummies(df['marital_status'], prefix='marital')
    
    df = pd.concat([df, education_dummies, marital_dummies], axis=1)
    
    scaler = StandardScaler()
    df['credit_given_scaled'] = scaler.fit_transform(df[['credit_given']])
    
    df.drop(['gender', 'education', 'marital_status'], axis=1, inplace=True)
    
    return df

def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    history_path = os.path.join(data_dir, 'payment_history.csv')
    default_path = os.path.join(data_dir, 'payment_default.csv')
    
    payment_history, payment_default = load_data(history_path, default_path)
    processed_default = preprocess_default_data(payment_default)
    
    processed_path = os.path.join(data_dir, 'processed_default.csv')
    save_processed_data(processed_default, processed_path)
