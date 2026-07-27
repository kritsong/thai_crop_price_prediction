import glob
import os
import numpy as np
import pandas as pd
from src.data.loader import CropDataLoader, EmptySeriesError

def build_dataset():
    """Build a pure autoregressive TFT dataset.

    The returned dataframe intentionally contains no exogenous columns. Calendar
    fields are retained as known future covariates because they are deterministic
    from the forecast date and do not leak price information.
    """
    loader = CropDataLoader()
    from paths import DATA_GLOB
    json_pattern = DATA_GLOB
    filepaths = glob.glob(json_pattern)
    print(f"Found {len(filepaths)} files.")

    # optional cap for fast smoke tests (0 = no cap, use all products)
    max_products = int(os.environ.get("TFT_MAX_PRODUCTS", "0"))

    dfs = []
    processed = 0
    for fp in filepaths:
        if max_products and processed >= max_products:
            break
        try:
            df = loader.load_and_preprocess(fp)
            if len(df) >= 70:
                # Add time index per group
                df['time_idx'] = np.arange(len(df))
                df['month'] = pd.to_datetime(df['date']).dt.month.astype(float)
                df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek.astype(float)
                df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear.astype(float)

                # We will use price_min as target
                if df['price_min'].isna().any():
                    df['price_min'] = df['price_min'].ffill().bfill()
                
                dfs.append(df)
                processed += 1
        except EmptySeriesError:
            pass
        except Exception:
            pass
            
    print(f"Successfully processed {processed} active products.")
    full_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    
    # TFT requires categorical types
    for col in ['product_id', 'category_name', 'group_name']:
        full_df[col] = full_df[col].astype(str).astype('category')
        
    return full_df, []
