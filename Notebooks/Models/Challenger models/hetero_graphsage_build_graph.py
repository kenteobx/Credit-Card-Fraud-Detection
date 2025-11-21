"""
Heterogeneous GraphSAGE – Script 1: Graph Construction & Final Processing

Upstream Stage (Stage 0: `feature_eng.ipynb`)
---------------------------------------------
This script requires the following four CSV files produced from the upstream stage:
    - train_X.csv  : features for the 80% Challenger train split
    - train_y.csv  : labels for the 80% Challenger train split
    - test_X.csv   : features for the 20% Challenger test split
    - test_y.csv   : labels for the 20% Challenger test split

Output
------
The script writes the graph to:
    graph_15features_scaled_balanced.pt

This file contains the fully balanced, feature-complete heterogeneous graph and is used
directly as the input dataset by the next stages:
    - `hetero_graphsage_model_training.ipynb`
    - `graphsage_lgbm_ensemble.ipynb`
"""




import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import Linear
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import polars as pl
import sys
import math 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder 
import gc 
import random

# Configuration
DATA_DIR = "/home/ubuntu/data/" # edit accordingly
train_x_path = os.path.join(DATA_DIR, "train_X.csv")
train_y_path = os.path.join(DATA_DIR, "train_y.csv")
test_x_path  = os.path.join(DATA_DIR, "test_X.csv")
test_y_path  = os.path.join(DATA_DIR, "test_y.csv")
# output path
save_path    = os.path.join(DATA_DIR, "graph_15features_scaled_balanced.pt")
# projection dimension for node features
PROJECTION_DIM = 128 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Helper functions
def pct_last(lst):
    """
    Percentile-of-last: proportion of values in lst that are <= last element.

    Used to measure how extreme the current transaction amount is relative
    to the card's recent 30-day amount history.
    """
    if not lst: return 0.0
    last = lst[-1]
    cnt_le = sum(1 for v in lst if v <= last)
    return float(cnt_le) / float(len(lst))

# Utility: Feature scaling
def scale_features(df, numeric_cols):
    """
    Scales numeric columns in a Polars DataFrame using Min-Max Scaling.

    Note:
    - Only the columns listed in numeric_cols are scaled.
    - The MinMaxScaler is fit on the full balanced dataset.
    - All other columns (IDs, categorical, labels) unchanged.
    """
    df_to_scale = df.select(numeric_cols)
    data_np = df_to_scale.to_numpy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_np)
    scaled_df = pl.DataFrame(scaled_data, schema=numeric_cols)
    df = df.drop(numeric_cols).hstack(scaled_df)
    return df
    
# Main: Build Heterogeneous Graph
def create_hetero_data(train_x_path, train_y_path, test_x_path, test_y_path):
    # Step 1: Loading and Tagging Raw Data
    print("Step 1: Loading raw data with Polars...")
    try:
        train_X = pl.read_csv(train_x_path).with_row_index('temp_idx')
        test_X = pl.read_csv(test_x_path).with_row_index('temp_idx')
        train_y = pl.read_csv(train_y_path).with_row_index('temp_idx')
        test_y = pl.read_csv(test_y_path).with_row_index('temp_idx')
    except Exception as e:
        print(f"Error: {e}. Make sure your _X.csv and _y.csv files are in {DATA_DIR}.")
        sys.exit(1)

    # Join X and y on temporary index (per provided split)
    train_df = train_X.join(train_y, on='temp_idx', how='inner')
    test_df = test_X.join(test_y, on='temp_idx', how='inner')
    
    # Replace temp_idx with a stable, global row index full_idx
    # train and test ranges are disjoint (no overlap in full_idx)
    train_df = train_df.drop('temp_idx').with_row_index('full_idx')
    test_df = test_df.drop('temp_idx').with_row_index('full_idx', offset=len(train_df))
    
    # Step 2: Stratified 70/10/20 splitting
    print("\nStep 2: Creating STRATIFIED masks (70/10/20 Split)...")
    
    # Treat provided train_df as the 80% (train+val) pool
    train_val_pool_df = train_df.to_pandas()
    
    # Stratified Split the 80% Pool (Train/Val) into 7:1, 70% train, 10% val
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(
        train_val_pool_df['full_idx'].to_numpy(), 
        train_val_pool_df['Is Fraud?'].to_numpy(), 
        test_size=0.125, # 10% / 80%
        random_state=42, 
        stratify=train_val_pool_df['Is Fraud?'].to_numpy()
    )
    # Provided test_df stays as the 20% test split
    test_indices = test_df['full_idx'].to_numpy()
    
    # Step 3: Balance train/val/test (undersample non-fraud)
    print("\nStep 3: Balancing all three sets...")
    target_ratio = 0.10 # Target fraud share (10%)
    
    def get_balanced_indices(df_to_sample_from, indices_pool, target_ratio=0.10, seed=42):
        """
        From a pool of row indices, undersample non-fraud rows so that:
            fraud_ratio approximately = target_ratio
        on the selected subset. The function returns the subset of `full_idx`
        to keep for that split.
        """
        unbalanced_df = df_to_sample_from.filter(pl.col('full_idx').is_in(indices_pool))
        n_fraud = unbalanced_df.filter(pl.col("Is Fraud?") == 1).height
        n_nonfraud = unbalanced_df.filter(pl.col("Is Fraud?") == 0).height
        
        if n_fraud == 0 or n_nonfraud == 0:
            print(f"Warning: Set has {n_fraud} frauds and {n_nonfraud} non-frauds. Cannot balance.")
            return unbalanced_df['full_idx'].to_numpy()
            
        # Solve for target non-fraud count given desired fraud share
        # target_ratio = n_fraud / (n_fraud + n_nonfraud_target)
        n_nonfraud_target = int(n_fraud * (1 - target_ratio) / target_ratio)
        frac = min(1.0, n_nonfraud_target / n_nonfraud)
        
        print(f"Balancing set: {n_fraud} frauds, {n_nonfraud} non-frauds. Sampling fraction: {frac:.4f}")
        fraud_df = unbalanced_df.filter(pl.col("Is Fraud?") == 1)
        nonfraud_sampled_df = unbalanced_df.filter(pl.col("Is Fraud?") == 0).sample(fraction=frac, seed=seed)
        
        return pl.concat([fraud_df, nonfraud_sampled_df])['full_idx'].to_numpy()

    # Use the original Polars splits for balancing; they still contain all rows
    print("Balancing Train Set (70%)...")
    kept_train_indices = get_balanced_indices(train_df, X_train_idx, target_ratio)
    
    print("Balancing Validation Set (10%)...")
    kept_val_indices = get_balanced_indices(train_df, X_val_idx, target_ratio)
    
    print("Balancing Test Set (20%)...")
    kept_test_indices = get_balanced_indices(test_df, test_indices, target_ratio)
    
    # Step 4: Build final balanced dataframe
    all_kept_indices = np.concatenate([kept_train_indices, kept_val_indices, kept_test_indices])
    
    full_df = pl.concat([train_df, test_df], how="vertical_relaxed")
    full_df = full_df.filter(pl.col('full_idx').is_in(all_kept_indices))
    
    print(f"\nTotal Edges in final balanced graph: {len(full_df)}")
    
    del train_df, test_df, train_val_pool_df
    gc.collect()

    # Step 5: Feature engineering (on balanced full_df)
    print("\nConverting Date and Time to DateTime object...")
    full_df = full_df.with_columns([
        pl.date(pl.col("Year"), pl.col("Month"), pl.col("Day")).alias("Date_dt")
    ])
    full_df = full_df.with_columns([
        (pl.col("Date_dt").cast(pl.Utf8) + " " + pl.col("Time").cast(pl.Utf8)).alias("DateTime_str")
    ])
    full_df = full_df.with_columns([
        pl.col("DateTime_str").str.to_datetime("%Y-%m-%d %H:%M", strict=False).alias("DateTime")
    ]).drop(["DateTime_str", "Date_dt"])
    
    print("\nStarting per-transaction Feature Engineering...")
    
    # IDs for per-card processing and daily frequency counts
    full_df = full_df.with_columns([(pl.col("User").cast(pl.Utf8) + "_" + pl.col("Card").cast(pl.Utf8)).alias("User_card"),])
    full_df = full_df.with_columns([
        (pl.col("User").cast(pl.Utf8) + "_" + pl.col("Date").cast(pl.Utf8)).alias("User_Date"),
        (pl.col("Merchant Name").cast(pl.Utf8) + "_" + pl.col("Date").cast(pl.Utf8)).alias("Merchant_Date"),
    ])
    full_df = full_df.with_columns([
        pl.col("User_Date").count().over("User_Date").alias("User_Freq_Day"),
        pl.col("Merchant_Date").count().over("Merchant_Date").alias("Merchant_Freq_Day"),
    ])

    # Clean and cast Amount to Float64
    full_df = full_df.with_columns([
        pl.col("Amount").cast(pl.Utf8)
            .str.replace_all(r"[^0-9\.\-]", "")
            .replace('', None)
            .cast(pl.Float64)
            .alias("Amount")
    ])

    print("Starting Per-Card Rolling Feature Loop...")
    cards = full_df["User_card"].unique().to_list()
    processed_frames = []

    for i, card in enumerate(cards):
        if (i % 500) == 0:
            print(f"Processing card {i}/{len(cards)}")

        # All transactions for this card in chronological order
        g = full_df.filter(pl.col("User_card") == card).sort("DateTime")
        if g.height == 0: continue

        # 1-hour rolling window (transaction intensity)
        rolled = g.rolling(index_column="DateTime", period="1h").agg([
            pl.col("Amount").sum().alias("txn_sum_1h"),
            pl.col("Amount").count().alias("txn_count_1h"),
        ])
        g = g.join(rolled, on="DateTime", how="left")
        g = g.with_columns([
            pl.col("txn_count_1h").fill_null(0).cast(pl.Int64),
            pl.col("txn_sum_1h").fill_null(0.0),
        ])

        # 30-day rolling statistics on amount
        rolled_30d = (
            g.rolling(index_column="DateTime", period="30d")
            .agg([
                pl.col("Amount").mean().alias("amount_mean_30d"),
                pl.col("Amount").std().alias("amount_std_30d"),
                pl.col("Amount").alias("amount_window_30d_list"),
            ])
        )
        g = g.join(rolled_30d, on="DateTime", how="left")

        # Percentile of current amount within the 30-day window
        lists = g["amount_window_30d_list"].to_list()
        percentiles = [pct_last(lst if lst is not None else []) for lst in lists]
        g = g.with_columns([pl.Series("amount_percentile_card_30d", percentiles)])

        # Z-score of current amount within the 30-day window
        def _zscore_row(s):
            amt = s["Amount"]
            mean = s["amount_mean_30d"]
            std = s["amount_std_30d"]
            if std is None or std == 0: return 0.0
            try: return float((amt - mean) / std)
            except Exception: return 0.0
        g = g.with_columns([
            pl.struct(["Amount", "amount_mean_30d", "amount_std_30d"]).map_elements(_zscore_row).alias("amount_zscore_30d")
        ])

        # Country / state change flag (merchant location shifts)
        g = g.with_columns([pl.col("Merchant State").shift(1).alias("prev_merchant_state")])
        g = g.with_columns([
            (pl.col("Merchant State") != pl.col("prev_merchant_state")).cast(pl.Int8).fill_null(0).alias("country_change_flag")
        ])

        # Merchant novelty for this card (first time seeing this merchant)
        g = g.with_columns([pl.col("Merchant Name").cum_count().alias("merchant_cumcount_for_card")])
        g = g.with_columns([
            (pl.col("merchant_cumcount_for_card") == 0).cast(pl.Int8).alias("is_new_merchant_for_card")
        ])
        
        # Drop intermediate helper columns
        drop_cols = ["card_mu", "card_R", "merchant_cumcount_for_card", "prev_merchant_state", "sec_of_day", "amount_window_30d_list"]
        g = g.drop([col for col in drop_cols if col in g.columns])
        processed_frames.append(g)
    
    full_df = pl.concat(processed_frames, how="vertical")
    del processed_frames
    gc.collect()

    # Drop unused / original raw columns that are now encoded elsewhere
    cols_to_drop = ["Time", "User_card", "card_kappa", "hour_angle", "User_Date", "Merchant_Date", "DayOfWeek", "Date"]
    full_df = full_df.drop([col for col in cols_to_drop if col in full_df.columns])

    # Step 5b: Label encoding
    df_pd = full_df.to_pandas()
    del full_df # save memory
    gc.collect()

    categorical_cols = ['Use Chip', 'Merchant City', 'Merchant State', 'Errors?', 'Zip_str']
    for col in categorical_cols:
        if col in df_pd.columns:
            if col == 'Zip_str' and 'Zip_str' not in df_pd.columns:
                if 'Zip' in df_pd.columns: col = 'Zip' 
                else: continue 
            le = LabelEncoder()
            df_pd[col] = le.fit_transform(df_pd[col].astype(str).fillna('Missing'))
    
    full_df = pl.from_pandas(df_pd)
    del df_pd
    gc.collect()
    
    print("--- Feature Engineering Complete ---")

    # Step 6: Feature definitions (for nodes and edges)
    merchant_cat_cols = ['Merchant City', 'Merchant State', 'MCC']
    user_cat_cols = ['Use Chip', 'Zip', 'Errors?']

    # These are the main numeric transaction features used for both edge_attr
    # and as aggregated features for the user node.
    full_transaction_features = [
        'Amount',
        'von_mises_likelihood_card',   
        'amount_log',
        'amount_mean_30d',
        'amount_std_30d',
        'amount_percentile_card_30d',
        'amount_zscore_30d',
        'hour_sin',
        'hour_cos',
        'country_change_flag',
        'is_new_merchant_for_card',
        'User_Freq_Day',
        'Merchant_Freq_Day',
        'txn_sum_1h',
        'txn_count_1h'
        ]

    user_node_num_cols = full_transaction_features
    edge_feature_cols = full_transaction_features
    
    # Imputation (numeric)
    print("\nCRITICAL: Applying Zero Imputation to Numeric Features...")
    valid_numeric_cols = [col for col in full_transaction_features if col in full_df.columns]
    print(f"Found {len(valid_numeric_cols)} of {len(full_transaction_features)} numeric features.")
    full_df = full_df.with_columns([pl.col(col).fill_null(0.0) for col in valid_numeric_cols])
    
    # Scaling (numeric)
    print("\nCRITICAL: Applying MinMax Feature Scaling...")
    full_df = scale_features(full_df, valid_numeric_cols)
    
    # Step 7: Node ID mapping (users/merchants)
    print("Step 7: Encoding user and merchant node indices...")
    users = full_df.select('User').unique().with_row_index('src_idx')
    full_df = full_df.join(users, on='User')
    num_users = len(users)
    merchants = full_df.select('Merchant Name').unique().with_row_index('dst_idx')
    full_df = full_df.join(merchants, on='Merchant Name')
    num_merchants = len(merchants)
    print(f"Found {num_users} unique users and {num_merchants} unique merchants.")

    # Step 8: Node feature aggregation
    print("Step 8a: Aggregating merchant features...")
    valid_merchant_cat_cols = [col for col in merchant_cat_cols if col in full_df.columns]
    merchant_df = full_df.select(['dst_idx'] + valid_merchant_cat_cols).unique(subset=['dst_idx'])
    if valid_merchant_cat_cols:
        merchant_features_df = merchant_df.to_dummies(columns=valid_merchant_cat_cols, drop_first=False)
    else:
        merchant_features_df = merchant_df
    
    # Ensure every merchant ID has a row (fill missing with zeros)
    all_merchant_ids = pl.DataFrame({'dst_idx': range(num_merchants)})
    merchant_features_agg = all_merchant_ids.join(merchant_features_df, on='dst_idx', how='left').fill_null(0)
    X_merchant = torch.tensor(merchant_features_agg.drop('dst_idx').to_numpy(), dtype=torch.float)
    print(f"Merchant feature matrix shape: {X_merchant.shape}")
    del merchant_df, merchant_features_df, merchant_features_agg

    print("Step 8b: Aggregating user features...")
    valid_user_num_cols = [col for col in user_node_num_cols if col in full_df.columns]
    valid_user_cat_cols = [col for col in user_cat_cols if col in full_df.columns]

    # Numeric: average over all transactions for each user
    user_num_agg = full_df.group_by('src_idx').agg([pl.mean(col).alias(col) for col in valid_user_num_cols])

    # Categorical: one-hot, then mean over user to frequency encoding
    if valid_user_cat_cols:
        user_cat_agg = full_df.select(['src_idx'] + valid_user_cat_cols).to_dummies(columns=valid_user_cat_cols, drop_first=False).group_by('src_idx').mean()
        user_features_agg = user_num_agg.join(user_cat_agg, on='src_idx', how='left')
    else:
        user_features_agg = user_num_agg
    all_user_ids = pl.DataFrame({'src_idx': range(num_users)})
    user_features_agg = all_user_ids.join(user_features_agg, on='src_idx', how='left').fill_null(0)
    X_user = torch.tensor(user_features_agg.drop('src_idx').to_numpy(), dtype=torch.float)
    print(f"User feature matrix shape: {X_user.shape}")
    del user_num_agg, user_features_agg
    if 'user_cat_agg' in locals():
        del user_cat_agg

    # Step 9: Edge index, labels, and edge attributes
    print("Step 9a: Creating edge index and labels...")
    src_tensor = torch.tensor(full_df['src_idx'].to_numpy(), dtype=torch.long)
    dst_tensor = torch.tensor(full_df['dst_idx'].to_numpy(), dtype=torch.long)
    edge_index = torch.stack([src_tensor, dst_tensor], dim=0)

    # Edge-level labels: Is Fraud? (binary classification)
    edge_label = torch.tensor(full_df['Is Fraud?'].to_numpy(), dtype=torch.float)
    
    print("Step 9b: Creating edge_attr matrix from transaction features...")
    valid_edge_feature_cols = [col for col in edge_feature_cols if col in full_df.columns]
    print(f"Creating edge_attr with {len(valid_edge_feature_cols)} features.")
    edge_attr = torch.tensor(full_df[valid_edge_feature_cols].fill_null(0).to_numpy(), dtype=torch.float)
    print(f"Edge feature matrix shape: {edge_attr.shape}")

    # Step 10: Train/val/test masks (edge-level)
    print("\nStep 10: Creating final balanced masks...")
    
    # Assign a continuous index for each row in the final balanced dataframe
    full_df = full_df.with_row_index('final_graph_idx')
    
    # Map original full_idx membership to the new final_graph_idx space
    train_mask = torch.zeros(len(full_df), dtype=torch.bool)
    val_mask = torch.zeros(len(full_df), dtype=torch.bool)
    test_mask = torch.zeros(len(full_df), dtype=torch.bool)
    
    train_indices_new = full_df.filter(pl.col('full_idx').is_in(kept_train_indices))['final_graph_idx'].to_numpy()
    val_indices_new = full_df.filter(pl.col('full_idx').is_in(kept_val_indices))['final_graph_idx'].to_numpy()
    test_indices_new = full_df.filter(pl.col('full_idx').is_in(kept_test_indices))['final_graph_idx'].to_numpy()

    train_mask[train_indices_new] = True
    val_mask[val_indices_new] = True
    test_mask[test_indices_new] = True

    print(f"Total Edges in final graph: {len(full_df)}")
    print(f"Train Edges (Balanced): {train_mask.sum().item()}")
    print(f"Val Edges (Balanced):   {val_mask.sum().item()}")
    print(f"Test Edges (Balanced):  {test_mask.sum().item()}")
    
    
    # Step 11: Assemble HeteroData
    print("Step 11: Assembling HeteroData object...")

    data = HeteroData()
    data['user'].x = X_user
    data['merchant'].x = X_merchant
    edge_type = ('user', 'transaction', 'merchant')
    data[edge_type].edge_index = edge_index
    data[edge_type].edge_label = edge_label
    data[edge_type].train_mask = train_mask
    data[edge_type].val_mask = val_mask
    data[edge_type].test_mask = test_mask
    data[edge_type].edge_attr = edge_attr
    
    return data


def main():
    # Set device, only used for the projection step
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # create_hetero_data now returns the data with balanced masks and engineered features
    data = create_hetero_data(
        train_x_path, train_y_path, test_x_path, test_y_path
    )
    
    if data is None:
        print("Data creation failed. Exiting.")
        sys.exit(1)

    print("\n--- Original Graph (stats) ---")
    print(data)

    # Step 12: Random linear projection of node features
    print("\nStarting feature projection to lower dimension...")
    set_seed(42) # to make Linear projection stable

    # Move features to device for projection
    user_x = data['user'].x.to(device)
    merchant_x = data['merchant'].x.to(device)
 
    user_proj = Linear(user_x.shape[1], PROJECTION_DIM).to(device)
    merchant_proj = Linear(merchant_x.shape[1], PROJECTION_DIM).to(device)

    with torch.no_grad():
        user_x_proj = user_proj(user_x).cpu()
        merchant_x_proj = merchant_proj(merchant_x).cpu()

    print(f"Projected user features shape: {user_x_proj.shape}")
    print(f"Projected merchant features shape: {merchant_x_proj.shape}")

    # Step 13: Create a compact HeteroData with projected x
    data_proj = HeteroData()
    data_proj['user'].x = user_x_proj
    data_proj['merchant'].x = merchant_x_proj

    edge_type = ('user', 'transaction', 'merchant')
    data_proj[edge_type].edge_index = data[edge_type].edge_index
    data_proj[edge_type].edge_label = data[edge_type].edge_label
    data_proj[edge_type].train_mask = data[edge_type].train_mask
    data_proj[edge_type].val_mask = data[edge_type].val_mask
    data_proj[edge_type].test_mask = data[edge_type].test_mask
    data_proj[edge_type].edge_attr = data[edge_type].edge_attr

    print("\n--- New Projected Graph (stats) ---")
    print(data_proj) 

    # Save the small graph to disk
    torch.save(data_proj, save_path)

    print(f"\nSUCCESS!")
    print(f"New, small graph saved to: {save_path}")
    print("You can now run the training script (train.py).")

if __name__ == "__main__":
    main()