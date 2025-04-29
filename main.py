import os
import argparse
import pandas as pd
from src.data_preprocessing import load_data, preprocess_default_data, save_processed_data
from src.feature_engineering import engineer_payment_features
from src.model_training import prepare_data_for_modeling, train_models
from src.model_evaluation import evaluate_model
from src.scoring_function import score_payment_default

def main(args):
    """
    Main function to run the entire pipeline
    """
    # Set up directories
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, 'data')
    models_dir = os.path.join(project_dir, 'models')
    output_dir = os.path.join(project_dir, 'output')
    
    # Create directories if they don't exist
    for directory in [data_dir, models_dir, output_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Define file paths
    history_path = args.history_file or os.path.join(data_dir, 'payment_history.csv')
    default_path = args.default_file or os.path.join(data_dir, 'payment_default.csv')
    processed_path = os.path.join(data_dir, 'processed_default.csv')
    features_path = os.path.join(data_dir, 'payment_history_features.csv')
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    payment_history, payment_default = load_data(history_path, default_path)
    
    # Step 2: Preprocess default data
    print("Step 2: Preprocessing default data...")
    processed_default = preprocess_default_data(payment_default)
    save_processed_data(processed_default, processed_path)
    
    # Step 3: Engineer features from payment history
    print("Step 3: Engineering features...")
    history_features = engineer_payment_features(payment_history)
    history_features.to_csv(features_path, index=False)
    
    # Step 4: Prepare data for modeling
    print("Step 4: Preparing data for modeling...")
    X, y = prepare_data_for_modeling(processed_default)
    
    # Step 5: Train models
    print("Step 5: Training models...")
    results, best_model = train_models(X, y, models_dir)
    
    # Step 6: Evaluate model
    print("Step 6: Evaluating model...")
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    model_path = os.path.join(models_dir, 'default_prediction_model.pkl')
    evaluation_metrics = evaluate_model(best_model, processed_default, pd.read_pickle(scaler_path), output_dir)
    
    # Step 7: Generate predictions
    print("Step 7: Generating predictions...")
    predictions = score_payment_default(history_path, default_path, models_dir)
    predictions.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    print(f"Pipeline completed successfully. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Payment Default Prediction Pipeline')
    parser.add_argument('--history_file', type=str, help='Path to payment history CSV file')
    parser.add_argument('--default_file', type=str, help='Path to payment default CSV file')
    
    args = parser.parse_args()
    main(args)