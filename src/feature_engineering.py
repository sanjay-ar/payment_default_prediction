import pandas as pd
import numpy as np

def engineer_payment_features(payment_df):
    df = payment_df.copy()
    
    df['payment_ratio'] = df['paid_amt'] / df['bill_amt'].replace(0, np.nan)
    df['payment_ratio'] = df['payment_ratio'].fillna(0)
    
    df['unpaid_amt'] = df['bill_amt'] - df['paid_amt']
    
    agg_features = df.groupby('client_id').agg({
        'payment_status': ['mean', 'min', 'max', 'std'],
        'bill_amt': ['sum', 'mean', 'max', 'min'],
        'paid_amt': ['sum', 'mean', 'max', 'min'],
        'payment_ratio': ['mean', 'min', 'max'],
        'unpaid_amt': ['sum', 'mean']
    }).reset_index()
    
    agg_features.columns = ['_'.join(col).strip('_') for col in agg_features.columns.values]
    
    df['on_time'] = (df['payment_status'] == -1).astype(int)
    ontime_rate = df.groupby('client_id')['on_time'].mean().reset_index()
    agg_features = pd.merge(agg_features, ontime_rate, on='client_id', how='left')
    
    for status in range(1, 10):
        col_name = f'late_{status}'
        df[col_name] = (df['payment_status'] == status).astype(int)
        status_count = df.groupby('client_id')[col_name].sum().reset_index()
        agg_features = pd.merge(agg_features, status_count, on='client_id', how='left')
    
    df['lateness_severity'] = df['payment_status'].apply(lambda x: max(0, x))
    severity_mean = df.groupby('client_id')['lateness_severity'].mean().reset_index()
    severity_max = df.groupby('client_id')['lateness_severity'].max().reset_index()
    
    agg_features = pd.merge(agg_features, severity_mean.rename(
        columns={'lateness_severity': 'lateness_severity_mean'}), on='client_id', how='left')
    agg_features = pd.merge(agg_features, severity_max.rename(
        columns={'lateness_severity': 'lateness_severity_max'}), on='client_id', how='left')
    
    agg_features = agg_features.fillna(0)
    
    return agg_features

def merge_features(client_df, payment_features_df):
    return pd.merge(client_df, payment_features_df, on='client_id', how='left').fillna(0)
