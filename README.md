# Payment Default Prediction System

This project implements a machine learning solution to predict payment defaults based on historical payment data and client information.

## Project Structure

```
payment_default_prediction/
├── data/              # Data files
├── notebooks/         # Jupyter notebooks
├── src/               # Source code
├── models/            # Trained models
├── output/            # Visualization and results
├── requirements.txt   # Dependencies
├── main.py            # Main execution script
└── README.md          # This file
```

## Setup Instructions

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Place your data files in the `data/` directory:
   - `payment_history.csv`: Historical payment data
   - `payment_default.csv`: Client information with default indicators

## Running the Pipeline

Execute the full pipeline with:

```
python main.py
```

