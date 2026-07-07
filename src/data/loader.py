import os
import json
import numpy as np
import pandas as pd

class EmptySeriesError(ValueError):
    """Exception raised when the loaded crop price series is empty or invalid."""
    pass

class CropDataLoader:
    def __init__(self, start_date="2018-01-03", end_date="2025-12-30"):
        """Sets up Monday-Friday business weekday skeleton dataframe between start_date and end_date."""
        self.start_date = start_date
        self.end_date = end_date
        all_business_days = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        self.skeleton = pd.DataFrame({'date': all_business_days.strftime('%Y-%m-%d')})

    def load_json(self, filepath):
        """Loads JSON using utf-8-sig encoding, extracts product_id, product_name, 
        category_name, group_name, and price_list. Raises EmptySeriesError if the 
        list is empty or if all prices (min and max) are zero or null. 
        Returns a raw dataframe and metadata dict.
        """
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            
        product_id = data.get('product_id', '')
        product_name = data.get('product_name', '')
        category_name = data.get('category_name', '')
        group_name = data.get('group_name', '')
        price_list = data.get('price_list', [])
        
        if not price_list:
            raise EmptySeriesError(f"price_list in {filepath} is empty")
            
        df_raw = pd.DataFrame(price_list)
        
        # Ensure target price columns exist
        for col in ['price_min', 'price_max']:
            if col not in df_raw.columns:
                df_raw[col] = np.nan
                
        # Check if all prices are zero or null
        vals = df_raw[['price_min', 'price_max']].values
        is_zero_or_null = (pd.isna(vals)) | (vals == 0) | (vals == 0.0)
        if is_zero_or_null.all():
            raise EmptySeriesError(f"All prices in {filepath} are zero or null")
            
        metadata = {
            'product_id': product_id,
            'product_name': product_name,
            'category_name': category_name,
            'group_name': group_name
        }
        
        return df_raw, metadata

    def clean_outliers(self, df, target_cols=['price_min', 'price_max']):
        """Sorts by parsed date, drops duplicates, cleans outliers on the raw reporting 
        dates using rolling 30-day IQR bounds: [rolling_median - 3 * IQR, rolling_median + 3 * IQR]. 
        If IQR is 0, use 0.01 * rolling_median as IQR. Replaces outliers with NaN, 
        then forward-fills and backward-fills. Vectorize this step for efficiency (no loops over rows!).
        """
        df = df.copy()
        
        # Ensure we have parsed date for sorting/dropping duplicates
        df['parsed_date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['parsed_date'])
        df = df.sort_values('parsed_date')
        df = df.drop_duplicates(subset=['parsed_date']).reset_index(drop=True)
        
        # Update date column as string representation
        df['date'] = df['parsed_date'].dt.strftime('%Y-%m-%d')
        df = df.drop(columns=['parsed_date'])
        
        for col in target_cols:
            if col not in df.columns:
                continue
                
            # Replace 0 with NaN so it can be forward-filled
            df.loc[df[col] == 0, col] = np.nan
            
            # Calculate rolling median and IQR
            rolling_median = df[col].rolling(window=30, min_periods=1).median()
            q1 = df[col].rolling(window=30, min_periods=1).quantile(0.25)
            q3 = df[col].rolling(window=30, min_periods=1).quantile(0.75)
            rolling_iqr = q3 - q1
            
            # If IQR is 0, use 0.01 * rolling_median as IQR
            adj_rolling_iqr = np.where(rolling_iqr == 0, 0.01 * rolling_median, rolling_iqr)
            
            lower_bound = rolling_median - 3 * adj_rolling_iqr
            upper_bound = rolling_median + 3 * adj_rolling_iqr
            
            # Replaces outliers with NaN
            is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
            df.loc[is_outlier, col] = np.nan
            
            # Forward-fills and backward-fills
            df[col] = df[col].ffill().bfill()
            
        return df

    def align_to_skeleton(self, df, target_cols=['price_min', 'price_max']):
        """Aligns to Monday-Friday business weekday skeleton via left merge, 
        forward-fills and backward-fills missing business days.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        df_aligned = pd.merge(self.skeleton, df, on='date', how='left')
        df_aligned = df_aligned.sort_values('date').reset_index(drop=True)
        
        for col in target_cols:
            if col in df_aligned.columns:
                df_aligned[col] = df_aligned[col].ffill().bfill()
                
        return df_aligned

    def load_and_preprocess(self, filepath):
        """Executes load_json, clean_outliers, and align_to_skeleton. 
        Then injects metadata columns ('product_id', 'group_name', 'category_name') into the dataframe.
        """
        df_raw, metadata = self.load_json(filepath)
        df_cleaned = self.clean_outliers(df_raw)
        df_aligned = self.align_to_skeleton(df_cleaned)
        
        # Volatility check: >= 1% of valid days must have a price change
        prices = df_aligned['price_max'].values.astype(float)
        prices_no_nan = prices[~np.isnan(prices)]
        if len(prices_no_nan) > 1:
            diffs = np.diff(prices_no_nan)
            n_changes = np.sum(diffs != 0)
            if len(diffs) > 0 and (n_changes / len(diffs)) < 0.01:
                raise EmptySeriesError(f"Product in {filepath} is too static (volatility < 1%)")
        elif len(prices_no_nan) <= 1:
            raise EmptySeriesError(f"Product in {filepath} has insufficient data points after cleaning")
            
        df_aligned['product_id'] = metadata.get('product_id', '')
        df_aligned['group_name'] = metadata.get('group_name', '')
        df_aligned['category_name'] = metadata.get('category_name', '')
        
        return df_aligned
