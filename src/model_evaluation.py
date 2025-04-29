import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

def load_model_and_data(model_path, data_path, scaler_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    data = pd.read_csv(data_path)
    
    return model, data, scaler

def evaluate_model(model, data, scaler, output_dir):
    X = data.drop(['client_id', 'month', 'default'], axis=1, errors='ignore')
    y = data['default']
    
    X_scaled = scaler.transform(X)
    
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Default Prediction Model')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X.columns
    else:
        try:
            importances = abs(model.coef_[0])
            feature_names = X.columns
        except:
            importances = None
            feature_names = None
    
    if importances is not None:
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    
    return {
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_dir, 'data')
    models_dir = os.path.join(project_dir, 'models')
    output_dir = os.path.join(project_dir, 'output')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(models_dir, 'default_prediction_model.pkl')
    data_path = os.path.join(data_dir, 'processed_default.csv')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    model, data, scaler = load_model_and_data(model_path, data_path, scaler_path)
    
    results = evaluate_model(model, data, scaler, output_dir)
    print("Model evaluation completed. Visualizations saved to output directory.")
