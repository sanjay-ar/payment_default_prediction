import pandas as pd
import numpy as np
import os
import pickle
from src.data_preprocessing import preprocess_default_data
from src.feature_engineering import engineer_payment_features, merge_features

def load_model_components(models_dir):
    model_path = os.path.join(models_dir, 'default_prediction_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    feature_names_path = os.path.join(models_dir, 'feature_names.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

def score_payment_default(payment_history_file, payment_default_file, models_dir):
    model, scaler, feature_names = load_model_components(models_dir)
    payment_history = pd.read_csv(payment_history_file)
    payment_default = pd.read_csv(payment_default_file)
    payment_history_features = engineer_payment_features(payment_history)
    payment_default_processed = preprocess_default_data(payment_default)
    final_data = payment_default_processed
    X_pred = final_data.drop(['client_id', 'month'], axis=1, errors='ignore')
    if 'default' in X_pred.columns:
        X_pred = X_pred.drop(['default'], axis=1)
    for feature in feature_names:
        if feature not in X_pred.columns:
            X_pred[feature] = 0
    X_pred = X_pred[feature_names]
    X_pred_scaled = scaler.transform(X_pred)
    default_probabilities = model.predict_proba(X_pred_scaled)[:, 1]
    default_indicators = model.predict(X_pred_scaled)
    results = pd.DataFrame({
        'client_id': final_data['client_id'],
        'default_probability': default_probabilities,
        'default_indicator': default_indicators
    })
    return results

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_dir, 'data')
    models_dir = os.path.join(project_dir, 'models')
    payment_history_file = os.path.join(data_dir, 'payment_history.csv')
    payment_default_file = os.path.join(data_dir, 'payment_default.csv')
    results = score_payment_default(payment_history_file, payment_default_file, models_dir)
    print(results.head())
    results.to_csv(os.path.join(data_dir, 'prediction_results.csv'), index=False)
    print(f"Prediction results saved to {os.path.join(data_dir, 'prediction_results.csv')}")
