import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb

def prepare_data_for_modeling(default_df):
    X = default_df.drop(['client_id', 'month', 'default'], axis=1, errors='ignore')
    if 'default' in default_df.columns:
        y = default_df['default']
    else:
        y = None
    return X, y

def train_models(X, y, save_dir):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': xgb.XGBClassifier(random_state=42)
    }
    best_model = None
    best_score = 0
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        results[name] = {
            'accuracy': accuracy,
            'auc': auc,
            'model': model
        }
        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        if auc > best_score:
            best_score = auc
            best_model = model
    best_model_path = os.path.join(save_dir, 'default_prediction_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to {best_model_path}")
    feature_names = X.columns.tolist()
    with open(os.path.join(save_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    return results, best_model

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_dir, 'data')
    models_dir = os.path.join(project_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    processed_path = os.path.join(data_dir, 'processed_default.csv')
    default_data = pd.read_csv(processed_path)
    X, y = prepare_data_for_modeling(default_data)
    results, best_model = train_models(X, y, models_dir)
