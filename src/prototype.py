import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_and_preprocess(filepath):
    print(f"Loading product data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
        
    p_id = data.get('product_id', '')
    p_name = data.get('product_name', '')
    price_list = data.get('price_list', [])
    
    df_p = pd.DataFrame(price_list)
    df_p['parsed_date'] = pd.to_datetime(df_p['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    df_p = df_p.dropna(subset=['parsed_date'])
    df_p = df_p.drop_duplicates(subset=['parsed_date']).sort_values('parsed_date').reset_index(drop=True)
    df_p = df_p[['parsed_date', 'price_min', 'price_max']].rename(columns={'parsed_date': 'date'})
    
    # Dynamic outlier detection and cleaning (similar to run_eda.py)
    for col in ['price_min', 'price_max']:
        rolling_median = df_p[col].rolling(window=30, min_periods=1).median()
        q1 = df_p[col].rolling(window=30, min_periods=1).quantile(0.25)
        q3 = df_p[col].rolling(window=30, min_periods=1).quantile(0.75)
        rolling_iqr = q3 - q1
        adj_rolling_iqr = np.where(rolling_iqr == 0, 0.01 * rolling_median, rolling_iqr)
        
        lower_bound = rolling_median - 3 * adj_rolling_iqr
        upper_bound = rolling_median + 3 * adj_rolling_iqr
        
        for idx, val in enumerate(df_p[col]):
            if pd.notna(val) and (val < lower_bound[idx] or val > upper_bound[idx]):
                df_p.at[idx, col] = np.nan
        df_p[col] = df_p[col].ffill().bfill()
        
    # Align price data to business weekdays (2018-01-03 to 2025-12-30)
    print("Aligning to business weekdays skeleton...")
    all_business_days = pd.date_range(start="2018-01-03", end="2025-12-30", freq='B')
    df_skeleton = pd.DataFrame({'date': all_business_days.strftime('%Y-%m-%d')})
    
    df_aligned = pd.merge(df_skeleton, df_p, on='date', how='left')
    df_aligned = df_aligned.sort_values('date').reset_index(drop=True)
    df_aligned['price_min'] = df_aligned['price_min'].ffill().bfill()
    df_aligned['price_max'] = df_aligned['price_max'].ffill().bfill()
    
    return df_aligned, p_id, p_name

def generate_features(df):
    print("Generating price-only features...")
    # Shift to prevent target leakage
    df['price_shifted'] = df['price_min'].shift(1)
    
    # Autoregressive lags
    df['lag_1'] = df['price_min'].shift(1)
    df['lag_7'] = df['price_min'].shift(7)
    df['lag_28'] = df['price_min'].shift(28)
    df['lag_70'] = df['price_min'].shift(70)
    
    # Rolling window stats
    df['roll_mean_7'] = df['price_shifted'].rolling(window=7).mean()
    df['roll_std_7'] = df['price_shifted'].rolling(window=7).std()
    df['roll_mean_30'] = df['price_shifted'].rolling(window=30).mean()
    df['roll_std_30'] = df['price_shifted'].rolling(window=30).std()
    
    # Drop rows with NaN in features
    feature_cols = ['lag_1', 'lag_7', 'lag_28', 'lag_70', 'roll_mean_7', 'roll_std_7', 'roll_mean_30', 'roll_std_30']
    df_feat = df.dropna(subset=feature_cols + ['price_min']).copy()
    return df_feat, feature_cols

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def main():
    from paths import DATA_DIR
    filepath = str(DATA_DIR / 'P11001.json')
    df_aligned, p_id, p_name = load_and_preprocess(filepath)
    
    df_feat, feature_cols = generate_features(df_aligned)
    
    # Split train (2018-2023) and test (2024-2025)
    train_df = df_feat[(df_feat['date'] >= '2018-01-01') & (df_feat['date'] <= '2023-12-31')].copy()
    test_df = df_feat[(df_feat['date'] >= '2024-01-01') & (df_feat['date'] <= '2025-12-31')].copy()
    
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(train_df[feature_cols])
    y_train = scaler_y.fit_transform(train_df[['price_min']])
    
    X_test = scaler_X.transform(test_df[feature_cols])
    y_test = scaler_y.transform(test_df[['price_min']])
    
    # Set PyTorch GPU device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Device selected: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    class PriceMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x)
            
    model = PriceMLP(len(feature_cols)).to(device)
    
    # Convert data to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Train MLP model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training PyTorch model...")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Training Loss: {loss.item():.6f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t).cpu().numpy()
        preds = scaler_y.inverse_transform(preds_scaled)
        
    test_df['predicted_price_min'] = preds
    
    # Metrics
    mae = mean_absolute_error(test_df['price_min'], test_df['predicted_price_min'])
    rmse = np.sqrt(mean_squared_error(test_df['price_min'], test_df['predicted_price_min']))
    s_val = smape(test_df['price_min'].values, test_df['predicted_price_min'].values)
    
    print("\n--- Model Evaluation ---")
    print(f"Product ID: {p_id} ({p_name})")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test SMAPE: {s_val:.4f}%")
    print("Forecasting prototype completed successfully!")

if __name__ == '__main__':
    main()
