import pandas as pd
import numpy as np

class FeatureGenerator:
    def __init__(self, target_col='price_min', horizons=None):
        """Sets up features list and multi-scale horizons."""
        self.target_col = target_col
        self.horizons = horizons if horizons is not None else [1, 20, 60, 120, 250]
        self.features = [
            'lag_1', 'lag_5', 'lag_7', 'lag_10', 'lag_20', 'lag_28', 'lag_40',
            'lag_60', 'lag_70', 'lag_120', 'lag_125', 'lag_250',
            'diff_1', 'diff_5', 'diff_20', 'diff_60', 'diff_120', 'diff_250',
            'return_1', 'return_5', 'return_20', 'return_60', 'return_120', 'return_250',
            'roll_mean_7', 'roll_std_7', 'roll_mean_30', 'roll_std_30',
            'roll_median_7', 'roll_median_30',
            'roll_min_30', 'roll_max_30', 'roll_range_30', 'roll_cv_30',
            'roll_mean_60', 'roll_std_60', 'roll_mean_120', 'roll_std_120',
            'days_since_change', 'changes_20', 'changes_60', 'changes_250',
            'month', 'day_of_week', 'day_of_year',
            'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos'
        ]
        
        # Weather data removed

    def generate_features(self, df):
        """Generates price_shifted = target_col.shift(1). 
        Generates leakage-safe price-only lag, difference, return, rolling, flat-run, and calendar features.
        Drops all rows containing NaN in the target or any feature columns.
        """
        df = df.copy()
        
        # Check if target column exists
        if self.target_col not in df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in dataframe")
            
        target_series = df[self.target_col]
        price_shifted = target_series.shift(1)
        
        # Lags. Every lag is strictly positive, so no feature uses the contemporaneous target.
        for lag in [1, 5, 7, 10, 20, 28, 40, 60, 70, 120, 125, 250]:
            df[f'lag_{lag}'] = target_series.shift(lag)
        
        # Temporal Features
        date_parsed = pd.to_datetime(df['date'])
        df['month'] = date_parsed.dt.month
        df['day_of_week'] = date_parsed.dt.dayofweek
        df['day_of_year'] = date_parsed.dt.dayofyear
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5.0)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5.0)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 366.0)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 366.0)
        
        # Rolling window stats on shifted price
        df['roll_mean_7'] = price_shifted.rolling(window=7).mean()
        df['roll_std_7'] = price_shifted.rolling(window=7).std()
        df['roll_mean_30'] = price_shifted.rolling(window=30).mean()
        df['roll_std_30'] = price_shifted.rolling(window=30).std()
        df['roll_median_7'] = price_shifted.rolling(window=7).median()
        df['roll_median_30'] = price_shifted.rolling(window=30).median()
        df['roll_min_30'] = price_shifted.rolling(window=30).min()
        df['roll_max_30'] = price_shifted.rolling(window=30).max()
        df['roll_range_30'] = df['roll_max_30'] - df['roll_min_30']
        df['roll_cv_30'] = df['roll_std_30'] / df['roll_mean_30'].replace(0, np.nan)
        df['roll_mean_60'] = price_shifted.rolling(window=60).mean()
        df['roll_std_60'] = price_shifted.rolling(window=60).std()
        df['roll_mean_120'] = price_shifted.rolling(window=120).mean()
        df['roll_std_120'] = price_shifted.rolling(window=120).std()

        # Price movement features computed from the last known price path only.
        for lag in [1, 5, 20, 60, 120, 250]:
            df[f'diff_{lag}'] = price_shifted.diff(lag)
            denominator = price_shifted.shift(lag).replace(0, np.nan)
            df[f'return_{lag}'] = df[f'diff_{lag}'] / denominator

        prev_change = price_shifted.diff().abs().gt(1e-9).fillna(False)
        change_groups = prev_change.cumsum()
        df['days_since_change'] = prev_change.groupby(change_groups).cumcount()
        df['changes_20'] = prev_change.astype(float).rolling(window=20).sum()
        df['changes_60'] = prev_change.astype(float).rolling(window=60).sum()
        df['changes_250'] = prev_change.astype(float).rolling(window=250).sum()
        
        # Construct multi-step targets
        for h in self.horizons:
            df[f'target_{h}'] = target_series.shift(-(h-1))
        
        # Weather data merging removed
        
        # Drop rows containing NaN in the target or any feature columns
        df[self.features] = df[self.features].replace([np.inf, -np.inf], np.nan)
        cols_to_check = [self.target_col] + self.features
        df_feat = df.dropna(subset=cols_to_check).copy()
        
        return df_feat

    def split_train_test(self, df_feat):
        """Splits the feature dataframe into train and test sets using:
        - Train: 2018-01-03 to 2023-12-29
        - Test: 2024-01-01 to 2025-12-30
        Within each split, we re-compute target_1...target_7 to ensure zero leakage
        across the temporal split boundaries, and drop rows with NaN in features or targets.
        """
        df_feat = df_feat.copy()
        date_parsed = pd.to_datetime(df_feat['date'])
        
        train_mask = (date_parsed >= pd.to_datetime('2018-01-03')) & (date_parsed <= pd.to_datetime('2023-12-29'))
        test_mask = (date_parsed >= pd.to_datetime('2024-01-01')) & (date_parsed <= pd.to_datetime('2025-12-30'))
        
        train_df = df_feat[train_mask].copy()
        test_df = df_feat[test_mask].copy()
        
        # Re-compute targets within each split to avoid leakage across splits
        for h in self.horizons:
            train_df[f'target_{h}'] = train_df[self.target_col].shift(-(h-1))
            test_df[f'target_{h}'] = test_df[self.target_col].shift(-(h-1))
            
        target_cols = [f'target_{h}' for h in self.horizons]
        cols_to_check = self.features + target_cols
        
        train_df = train_df.dropna(subset=cols_to_check).copy()
        test_df = test_df.dropna(subset=cols_to_check).copy()
        
        return train_df, test_df

    def split_train_val_test(self, df_feat):
        """Strict temporal split for model selection and final evaluation.

        Train targets are contained inside 2018-2022, validation targets inside 2023,
        and test targets inside 2024-2025. Features remain computed from the full
        historical path, so the first validation/test rows may use already-known
        prices from the previous split boundary, but labels never cross boundaries.
        """
        df_feat = df_feat.copy()
        date_parsed = pd.to_datetime(df_feat['date'])

        train_mask = (date_parsed >= pd.to_datetime('2018-01-03')) & (date_parsed <= pd.to_datetime('2022-12-30'))
        val_mask = (date_parsed >= pd.to_datetime('2023-01-02')) & (date_parsed <= pd.to_datetime('2023-12-29'))
        test_mask = (date_parsed >= pd.to_datetime('2024-01-01')) & (date_parsed <= pd.to_datetime('2025-12-30'))

        splits = []
        target_cols = [f'target_{h}' for h in self.horizons]
        cols_to_check = self.features + target_cols

        for mask in [train_mask, val_mask, test_mask]:
            split_df = df_feat[mask].copy()
            for h in self.horizons:
                split_df[f'target_{h}'] = split_df[self.target_col].shift(-(h - 1))
            split_df = split_df.dropna(subset=cols_to_check).copy()
            splits.append(split_df)

        return tuple(splits)
