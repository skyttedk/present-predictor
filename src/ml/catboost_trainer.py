# src/ml/catboost_trainer.py

"""
CatBoost Model Training Script

This script implements the CatBoost Regressor model training pipeline,
migrated from the 'notebooks/catboost_implementation.ipynb' notebook.
It includes data loading, preprocessing, feature engineering,
CatBoost model training with RMSE loss, hyperparameter tuning with Optuna,
cross-validation, and artifact saving.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import pickle
import joblib
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder # For y_strata
from sklearn.feature_extraction import FeatureHasher

import catboost
from catboost import CatBoostRegressor, Pool
import optuna

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Should point to project root
DATA_PATH = os.path.join(BASE_DIR, "src/data/historical/present.selection.historic.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models", "catboost_rmse_model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "catboost_rmse")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Logging Setup ---
LOG_FILE = os.path.join(LOGS_DIR, f"catboost_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info(f"CatBoost version: {catboost.__version__}")
logging.info("Libraries imported successfully.")
logging.info(f"Base directory resolved to: {BASE_DIR}")
logging.info(f"Data path: {DATA_PATH}")
logging.info(f"Models directory: {MODELS_DIR}")
logging.info(f"Reports directory: {REPORTS_DIR}")

def load_and_clean_data(data_path: str) -> pd.DataFrame:
    """Loads and cleans the raw historical data."""
    logging.info(f"[DATA] Loading historical selection data from {data_path}...")
    raw_data = pd.DataFrame()
    try:
        raw_data = pd.read_csv(data_path, encoding='utf-8', dtype='str')
        logging.info(f"Loaded {len(raw_data)} selection events.")
        logging.info(f"Columns: {list(raw_data.columns)}")
    except FileNotFoundError:
        logging.error(f"ERROR: {data_path} not found.")
        return pd.DataFrame()

    if raw_data.empty:
        logging.warning("Raw data is empty.")
        return pd.DataFrame()

    logging.info("[CLEAN] Cleaning data...")
    cleaned_data = raw_data.copy()
    for col in cleaned_data.columns:
        cleaned_data[col] = cleaned_data[col].astype(str).str.strip('"').str.strip()
    cleaned_data = cleaned_data.fillna("NONE")

    categorical_cols_to_lower = [
        'employee_gender', 'product_target_gender',
        'product_utility_type', 'product_durability', 'product_type'
    ]
    for col in categorical_cols_to_lower:
        if col in cleaned_data.columns:
            cleaned_data[col] = cleaned_data[col].str.lower()
            
    # Consolidate employee_shop and employee_branch: Use employee_branch as the shop identifier
    if 'employee_branch' in cleaned_data.columns and 'employee_shop' in cleaned_data.columns:
        logging.info("Consolidating 'employee_shop' to be identical to 'employee_branch'.")
        cleaned_data['employee_shop'] = cleaned_data['employee_branch']
    elif 'employee_branch' in cleaned_data.columns:
        logging.warning("'employee_shop' column not found. Creating it from 'employee_branch'.")
        cleaned_data['employee_shop'] = cleaned_data['employee_branch']
    elif 'employee_shop' in cleaned_data.columns:
        logging.warning("'employee_branch' column not found. Using 'employee_shop' as is, but this might be inconsistent with prediction logic if branch is expected.")
        # No change needed if only employee_shop exists, but this is less ideal.
    else:
        logging.error("Neither 'employee_shop' nor 'employee_branch' found. Cannot consolidate shop identifier.")
        # Potentially raise an error or return empty if this is critical
        
    logging.info(f"Data cleaning and shop/branch consolidation complete: {len(cleaned_data)} records")
    return cleaned_data

def aggregate_data(cleaned_data: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Aggregates data and calculates selection_rate."""
    if cleaned_data.empty:
        logging.warning("Cleaned data is empty, skipping aggregation.")
        return pd.DataFrame(), []

    grouping_cols = [
        'employee_shop', 'employee_branch', 'employee_gender',
        'product_main_category', 'product_sub_category', 'product_brand',
        'product_color', 'product_durability', 'product_target_gender',
        'product_utility_type', 'product_type'
    ]
    
    # Ensure total_employees_in_group is numeric
    if 'total_employees_in_group' not in cleaned_data.columns:
        logging.error("`total_employees_in_group` column not found. Cannot calculate selection rate.")
        raise ValueError("`total_employees_in_group` column not found.")
        
    cleaned_data['total_employees_in_group'] = pd.to_numeric(cleaned_data['total_employees_in_group'], errors='coerce').fillna(0)

    logging.info(f"[AGGREGATE] Grouping by {len(grouping_cols)} features: {grouping_cols}")
    
    agg_data = cleaned_data.groupby(grouping_cols).agg(
        selection_count=('employee_shop', 'size'),
        total_employees_in_group=('total_employees_in_group', 'first') # Assuming it's constant per group
    ).reset_index()

    # Calculate selection_rate
    agg_data['selection_rate'] = agg_data['selection_count'] / agg_data['total_employees_in_group']
    agg_data['selection_rate'] = agg_data['selection_rate'].fillna(0) # Handle division by zero
    
    if not agg_data.empty:
        compression_ratio = len(cleaned_data) / len(agg_data) if len(agg_data) > 0 else 0
        logging.info("Aggregation complete:")
        logging.info(f"  {len(cleaned_data)} events -> {len(agg_data)} unique combinations")
        logging.info(f"  Compression ratio: {compression_ratio:.1f}x")
    else:
        logging.warning("agg_data is empty after grouping.")
    return agg_data, grouping_cols

def engineer_shop_assortment_features(
    df_to_engineer: pd.DataFrame,
    lookup_source_df: pd.DataFrame,
    selection_count_col: str = 'selection_count'
) -> pd.DataFrame:
    """
    Engineers shop assortment features in a way that prevents data leakage.
    It computes aggregates from `lookup_source_df` (training set) and
    merges them into `df_to_engineer` (train or validation set).
    """
    logging.info("[FEATURE ENG] Engineering shop assortment features (leakage-proof)...")
    if df_to_engineer.empty:
        logging.warning("Skipping shop features as df_to_engineer is empty.")
        return df_to_engineer.copy()
    if lookup_source_df.empty:
        logging.warning("Skipping shop features as lookup_source_df is empty.")
        return df_to_engineer.copy()

    # 1. Create the feature lookup table from the source dataframe (training data)
    logging.info(f"Creating lookup table from source with shape {lookup_source_df.shape}")
    
    source_df = lookup_source_df.copy()
    source_df[selection_count_col] = pd.to_numeric(source_df[selection_count_col], errors='coerce').fillna(0)

    shop_summary = source_df.groupby('employee_shop').agg(
        unique_product_combinations_in_shop=('product_main_category', 'count'),
        distinct_main_categories_in_shop=('product_main_category', 'nunique'),
        distinct_sub_categories_in_shop=('product_sub_category', 'nunique'),
        distinct_brands_in_shop=('product_brand', 'nunique'),
        distinct_utility_types_in_shop=('product_utility_type', 'nunique')
    ).reset_index()

    shop_main_cat_counts = source_df.groupby(['employee_shop', 'product_main_category'])[selection_count_col].sum().reset_index()
    idx = shop_main_cat_counts.groupby(['employee_shop'])[selection_count_col].transform(max) == shop_main_cat_counts[selection_count_col]
    shop_top_main_cats = shop_main_cat_counts[idx].drop_duplicates(subset=['employee_shop'], keep='first')
    shop_top_main_cats = shop_top_main_cats[['employee_shop', 'product_main_category']].rename(
        columns={'product_main_category': 'shop_most_frequent_main_category_selected'}
    )

    shop_brand_counts = source_df.groupby(['employee_shop', 'product_brand'])[selection_count_col].sum().reset_index()
    idx_brand = shop_brand_counts.groupby(['employee_shop'])[selection_count_col].transform(max) == shop_brand_counts[selection_count_col]
    shop_top_brands = shop_brand_counts[idx_brand].drop_duplicates(subset=['employee_shop'], keep='first')
    shop_top_brands = shop_top_brands[['employee_shop', 'product_brand']].rename(
        columns={'product_brand': 'shop_most_frequent_brand_selected'}
    )

    shop_features_lookup = shop_summary.copy()
    if not shop_top_main_cats.empty:
        shop_features_lookup = pd.merge(shop_features_lookup, shop_top_main_cats, on='employee_shop', how='left')
    if not shop_top_brands.empty:
        shop_features_lookup = pd.merge(shop_features_lookup, shop_top_brands, on='employee_shop', how='left')

    shop_features_lookup = shop_features_lookup.rename(columns={
        'distinct_main_categories_in_shop': 'shop_main_category_diversity_selected',
        'distinct_brands_in_shop': 'shop_brand_diversity_selected',
        'distinct_utility_types_in_shop': 'shop_utility_type_diversity_selected',
        'distinct_sub_categories_in_shop': 'shop_sub_category_diversity_selected'
    })

    # 2. Merge lookup table into the target dataframe
    logging.info(f"Merging lookup table into target df with shape {df_to_engineer.shape}")
    df_with_features = pd.merge(df_to_engineer, shop_features_lookup, on='employee_shop', how='left')

    # 3. Handle NaNs for unseen shops in validation/test set
    numeric_shop_cols = [
        'unique_product_combinations_in_shop', 'shop_main_category_diversity_selected',
        'shop_brand_diversity_selected', 'shop_utility_type_diversity_selected',
        'shop_sub_category_diversity_selected'
    ]
    categorical_shop_cols = [
        'shop_most_frequent_main_category_selected', 'shop_most_frequent_brand_selected'
    ]

    for col in numeric_shop_cols:
        if col in df_with_features.columns and df_with_features[col].isnull().any():
            median_val = shop_features_lookup[col].median()
            logging.info(f"Filling NaNs in numeric shop feature '{col}' with median value: {median_val}")
            df_with_features[col] = df_with_features[col].fillna(median_val)

    for col in categorical_shop_cols:
        if col in df_with_features.columns and df_with_features[col].isnull().any():
            logging.info(f"Filling NaNs in categorical shop feature '{col}' with 'NONE'")
            df_with_features[col] = df_with_features[col].fillna("NONE")

    # Final check for numeric types
    for col in numeric_shop_cols:
        if col in df_with_features.columns:
            df_with_features[col] = pd.to_numeric(df_with_features[col], errors='coerce').fillna(0)

    # 4. Create derived features based on the merged data
    if 'shop_most_frequent_main_category_selected' in df_with_features.columns:
        df_with_features['is_shop_most_frequent_main_category'] = (
            df_with_features['product_main_category'] == df_with_features['shop_most_frequent_main_category_selected']
        ).astype(int)
    else:
        df_with_features['is_shop_most_frequent_main_category'] = 0

    if 'shop_most_frequent_brand_selected' in df_with_features.columns:
        df_with_features['is_shop_most_frequent_brand'] = (
            df_with_features['product_brand'] == df_with_features['shop_most_frequent_brand_selected']
        ).astype(int)
    else:
        df_with_features['is_shop_most_frequent_brand'] = 0
    
    logging.info(f"Shape after shop features: {df_with_features.shape}")
    return df_with_features

def engineer_new_interaction_features(df: pd.DataFrame, n_interaction_features: int = 32) -> pd.DataFrame:
    """Engineers new interaction features using FeatureHasher with multiple interaction sets."""
    logging.info(f"[FEATURE ENG] Engineering new interaction features (target: {n_interaction_features}). Initial shape: {df.shape}")
    if df.empty:
        logging.warning("Skipping new interaction feature engineering as input df is empty.")
        return df.copy()

    hasher = FeatureHasher(n_features=n_interaction_features, input_type='string')
    
    # Create multiple interaction sets for better signal capture
    # First set: shop x main_category
    interaction1 = df.apply(
        lambda x: [
            f"shop_{x.get('employee_shop', 'NONE')}",
            f"cat_{x.get('product_main_category', 'NONE')}"
        ],
        axis=1
    )
    
    # Second set: brand x target_gender
    interaction2 = df.apply(
        lambda x: [
            f"brand_{x.get('product_brand', 'NONE')}",
            f"gender_{x.get('product_target_gender', 'NONE')}"
        ],
        axis=1
    )
    
    # Third set: sub_category x utility_type
    interaction3 = df.apply(
        lambda x: [
            f"subcat_{x.get('product_sub_category', 'NONE')}",
            f"utility_{x.get('product_utility_type', 'NONE')}"
        ],
        axis=1
    )
    
    # Hash all interaction sets
    hash1 = hasher.transform(interaction1).toarray()
    hash2 = hasher.transform(interaction2).toarray()
    hash3 = hasher.transform(interaction3).toarray()
    
    # Combine all hashes
    all_hashes = np.hstack([hash1, hash2, hash3])
    
    # Add as columns
    for i in range(all_hashes.shape[1]):
        df[f'interaction_hash_{i}'] = all_hashes[:, i]
    
    logging.info(f"Shape after new interaction features: {df.shape}")
    return df


def engineer_non_leaky_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Engineers non-leaky features that don't use the target variable.
    Replaces the previous product_relativity_features that caused data leakage.
    """
    logging.info("[FEATURE ENG] Engineering non-leaky features...")
    if df.empty:
        logging.warning("Skipping feature engineering as df is empty.")
        return df.copy()

    # For training data, we can calculate mode-based features
    # For test data, we'll use placeholders to avoid leakage
    if is_training and 'selection_count' in df.columns:
        # Calculate most frequent category/brand per shop using mode (not using selection_count directly)
        shop_main_cat_mode = df.groupby('employee_shop')['product_main_category'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 'NONE'
        ).reset_index()
        shop_main_cat_mode.columns = ['employee_shop', 'shop_most_frequent_main_category_mode']
        
        shop_brand_mode = df.groupby('employee_shop')['product_brand'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 'NONE'
        ).reset_index()
        shop_brand_mode.columns = ['employee_shop', 'shop_most_frequent_brand_mode']
        
        # Merge back
        df = pd.merge(df, shop_main_cat_mode, on='employee_shop', how='left')
        df = pd.merge(df, shop_brand_mode, on='employee_shop', how='left')
        
        # Update the most frequent columns
        if 'shop_most_frequent_main_category_selected' in df.columns:
            df['shop_most_frequent_main_category_selected'] = df['shop_most_frequent_main_category_mode']
        if 'shop_most_frequent_brand_selected' in df.columns:
            df['shop_most_frequent_brand_selected'] = df['shop_most_frequent_brand_mode']
        
        # Drop temporary columns
        df = df.drop(columns=['shop_most_frequent_main_category_mode', 'shop_most_frequent_brand_mode'], errors='ignore')

    logging.info(f"Shape after non-leaky features: {df.shape}")
    logging.info("Removed leaked features: product_share_in_shop, brand_share_in_shop, product_rank_in_shop, brand_rank_in_shop")
    
    return df


def prepare_features_for_catboost(final_features_df: pd.DataFrame, grouping_cols: list) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list, dict]:
    """Prepares X, y, exposure, y_strata, identifies categorical features, and calculates numeric medians for CatBoost."""
    logging.info("[FEATURES PREP] Preparing final X, y, exposure for CatBoost and calculating medians...")
    if final_features_df.empty or 'selection_count' not in final_features_df.columns:
        logging.warning("Skipping final feature preparation as final_features_df is empty or 'selection_count' is missing.")
        return pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64'), [], {}

    # Use selection_count as target for Poisson objective
    y = pd.to_numeric(final_features_df['selection_count'], errors='coerce').fillna(0)
    # Use total_employees_in_group as exposure for Poisson
    exposure = pd.to_numeric(final_features_df['total_employees_in_group'], errors='coerce').fillna(1)
    X = final_features_df.drop(columns=['selection_count', 'selection_rate', 'total_employees_in_group']).copy()

    cat_feature_names = list(grouping_cols)
    cat_feature_names.extend([
        'shop_most_frequent_main_category_selected',
        'shop_most_frequent_brand_selected',
    ])

    valid_categorical_features = []
    for col in cat_feature_names:
        if col in X.columns:
            X[col] = X[col].astype(str) # Ensure they are string type for CatBoost
            valid_categorical_features.append(col)
        else:
            logging.warning(f"Categorical feature '{col}' intended for CatBoost not found in X.")
    
    # Calculate medians for numeric columns BEFORE filling NaNs
    numeric_medians = {}
    for col in X.columns:
        if col not in valid_categorical_features: # If it's supposed to be numeric
            # Attempt to convert to numeric if it's an object type
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            if pd.api.types.is_numeric_dtype(X[col]):
                median_val = X[col].median()
                if pd.isna(median_val):
                    # Determine appropriate default if median is NaN
                    default_for_nan_median = 99.0 if col.endswith('_rank_in_shop') else 0.0
                    numeric_medians[col] = default_for_nan_median
                    logging.warning(f"Median for numeric column '{col}' is NaN. Storing default median: {default_for_nan_median}.")
                else:
                    numeric_medians[col] = median_val
            else:
                # If still not numeric after coercion attempts, it's an issue.
                # Store a defined default and log a warning.
                default_for_non_numeric = 99.0 if col.endswith('_rank_in_shop') else 0.0
                numeric_medians[col] = default_for_non_numeric
                logging.warning(f"Column '{col}' is not reliably numeric after conversion attempts. Storing default median: {default_for_non_numeric}.")

    # Handle potential NaNs and ensure correct types
    for col in X.columns:
        if col not in valid_categorical_features: # Numeric column
            if X[col].isnull().any():
                X[col] = X[col].fillna(numeric_medians.get(col, 0)) # Fill NaNs with calculated median or 0 if median not found
            # Ensure it's numeric after filling
            if not pd.api.types.is_numeric_dtype(X[col]):
                 X[col] = pd.to_numeric(X[col], errors='coerce').fillna(numeric_medians.get(col, 0))


        elif X[col].isnull().any(): # Categorical column with NaNs
             X[col] = X[col].fillna("NONE") # CatBoost handles string 'NONE' in categorical features
    
    final_cat_feature_indices = [X.columns.get_loc(col) for col in valid_categorical_features if col in X.columns]

    # Update stratification for count target (selection_count)
    y_strata = pd.cut(y, bins=[-1, 0, 1, 2, 5, 10, np.inf], labels=[0, 1, 2, 3, 4, 5], include_lowest=True)

    logging.info(f"Final X features shape: {X.shape}")
    logging.info(f"Target y shape: {y.shape}")
    logging.info(f"Exposure shape: {exposure.shape}")
    logging.info(f"Categorical features for CatBoost ({len(valid_categorical_features)}): {valid_categorical_features}")
    logging.info(f"Categorical feature indices: {final_cat_feature_indices}")
    logging.info(f"Calculated numeric medians: {numeric_medians}")
    if not y_strata.empty:
      logging.info(f"Stratification distribution:\n{y_strata.value_counts().sort_index().to_dict()}")
    
    return X, y, exposure, y_strata, valid_categorical_features, numeric_medians


# --- Main Pipeline ---
def main():
    logging.info("--- Starting CatBoost Model Training Pipeline ---")

    # 1. Load and Clean Data
    cleaned_df = load_and_clean_data(DATA_PATH)
    if cleaned_df.empty:
        logging.error("Failed to load or clean data. Exiting.")
        return

    # 2. Aggregate Data
    agg_df, base_grouping_cols = aggregate_data(cleaned_df)
    if agg_df.empty:
        logging.error("Failed to aggregate data. Exiting.")
        return

    # 3. IMPORTANT: Split data BEFORE feature engineering to avoid leakage
    logging.info("[SPLIT] Creating train/test split BEFORE feature engineering...")
    
    # Prepare stratification on the aggregated data (use selection_count for Poisson)
    y_temp = pd.to_numeric(agg_df['selection_count'], errors='coerce').fillna(0)
    y_strata_temp = pd.cut(y_temp, bins=[-1, 0, 1, 2, 5, 10, np.inf], labels=[0, 1, 2, 3, 4, 5], include_lowest=True)
    
    # Split indices
    train_indices, val_indices = train_test_split(
        np.arange(len(agg_df)),
        test_size=0.2,
        random_state=42,
        stratify=y_strata_temp
    )
    
    train_df = agg_df.iloc[train_indices].copy()
    val_df = agg_df.iloc[val_indices].copy()
    
    logging.info(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

    # 4. Engineer features separately for train and validation
    logging.info("[FEATURE ENG] Engineering features separately for train and validation...")
    
    # Train features
    train_with_shop = engineer_shop_assortment_features(train_df, train_df)
    train_with_interactions = engineer_new_interaction_features(train_with_shop)
    train_final = engineer_non_leaky_features(train_with_interactions, is_training=True)
    
    # Validation features
    val_with_shop = engineer_shop_assortment_features(val_df, train_df)
    val_with_interactions = engineer_new_interaction_features(val_with_shop)
    val_final = engineer_non_leaky_features(val_with_interactions, is_training=False)

    # 5. Prepare Features for CatBoost
    X_train, y_train, exposure_train, _, cat_features_for_model, numeric_medians = prepare_features_for_catboost(train_final, base_grouping_cols)
    X_val, y_val, exposure_val, _, _, _ = prepare_features_for_catboost(val_final, base_grouping_cols)
    
    # Ensure validation features match training features using a definitive feature list
    final_feature_list = list(X_train.columns)
    X_val = X_val.reindex(columns=final_feature_list)

    # Fill any missing values in X_val that might have resulted from reindexing
    for col in final_feature_list:
        if X_val[col].isnull().any():
            if col in cat_features_for_model:
                X_val[col] = X_val[col].fillna("NONE")
            else:
                # Use the median from the training set for numeric features
                X_val[col] = X_val[col].fillna(numeric_medians.get(col, 0))
    
    if X_train.empty or y_train.empty:
        logging.error("Feature preparation failed. Exiting.")
        return

    logging.info("--- Data Preparation Complete ---")
    logging.info(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    logging.info(f"Num Cat Features: {len(cat_features_for_model)}")
    
    # 6. Train Initial CatBoost Model
    
    initial_model_params = {
        'iterations': 1000,
        'loss_function': 'Poisson',
        'eval_metric': 'Poisson',
        'random_seed': 42,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'early_stopping_rounds': 50,
        'verbose': 100
    }
    trained_initial_model, initial_metrics = train_catboost_model(
        X_train, y_train, X_val, y_val, exposure_train, exposure_val, cat_features_for_model, initial_model_params, "Initial"
    )

    if not trained_initial_model:
        logging.error("Initial model training failed. Exiting.")
        return

    # 7. Hyperparameter Tuning with Optuna
    best_params_optuna = tune_hyperparameters_optuna(
        X_train, y_train, X_val, y_val, exposure_train, exposure_val, cat_features_for_model, n_trials=300
    )
    
    if not best_params_optuna:
        logging.warning("Optuna tuning did not find best parameters, using initial model parameters for further steps.")
        best_params_final = initial_model_params
    else:
        best_params_final = best_params_optuna
        best_params_final['loss_function'] = 'Poisson'
        best_params_final['eval_metric'] = 'Poisson'
        best_params_final['random_seed'] = 42
        best_params_final['verbose'] = 100
        if 'early_stopping_rounds' not in best_params_final:
             best_params_final['early_stopping_rounds'] = initial_model_params.get('early_stopping_rounds', 50)

    # 8. Train Final Model with Best/Chosen Parameters
    trained_final_model, final_metrics_val = train_catboost_model(
        X_train, y_train, X_val, y_val, exposure_train, exposure_val, cat_features_for_model, best_params_final, "Final Tuned"
    )

    if not trained_final_model:
        logging.error("Final model training failed. Exiting.")
        return

    # 9. Cross-Validation with Final Parameters
    # Since we split early, we'll skip CV for now or could implement it differently
    cv_scores = []
    logging.info("[CROSS-VALIDATION] Skipping CV as data was split early for leak prevention")

    # 10. Feature Importance
    importance_df = analyze_and_save_feature_importance(
        trained_final_model, X_train.columns, os.path.join(REPORTS_DIR, "figures", "catboost_feature_importance.png")
    )

    # 11. Compute and Save Shop Resolver Aggregates (from train_df only)
    logging.info("[AGGREGATES] Computing and saving shop resolver aggregates from training data...")
    compute_and_save_shop_resolver_aggregates(train_df.copy(), MODELS_DIR, selection_count_col='selection_count') # Use original selection_count

    # 12. Save Model Artifacts
    save_model_artifacts(
        trained_final_model,
        best_params_final,
        {**final_metrics_val, "cv_r2_mean": np.mean(cv_scores) if cv_scores else None, "cv_r2_std": np.std(cv_scores) if cv_scores else None},
        list(X_train.columns),
        cat_features_for_model,
        MODELS_DIR,
        importance_df,
        numeric_medians # Pass medians to save_model_artifacts
    )

    logging.info("--- CatBoost Model Training Pipeline Finished ---")

def compute_and_save_shop_resolver_aggregates(train_df: pd.DataFrame, model_dir: str, selection_count_col: str = 'selection_count'):
    """
    Computes shop-level aggregates and product relativity features solely from the training data
    and saves them as artifacts for the ShopFeatureResolver.
    """
    logging.info(f"Starting computation of shop resolver aggregates from train_df with shape {train_df.shape}")

    if train_df.empty:
        logging.warning("train_df is empty. Skipping shop resolver aggregate computation.")
        return

    # Ensure selection_count_col is numeric
    train_df[selection_count_col] = pd.to_numeric(train_df[selection_count_col], errors='coerce').fillna(0)

    # 1. historical_features_snapshot (Shop diversities, most frequent items)
    historical_features_snapshot = {}
    shop_stats_agg = train_df.groupby('employee_shop').agg(
        shop_main_category_diversity_selected=('product_main_category', 'nunique'),
        shop_sub_category_diversity_selected=('product_sub_category', 'nunique'),
        shop_brand_diversity_selected=('product_brand', 'nunique'),
        shop_utility_type_diversity_selected=('product_utility_type', 'nunique')
    ).reset_index()

    shop_main_cats_counts = train_df.groupby(['employee_shop', 'product_main_category'])[selection_count_col].sum().reset_index(name='count')
    shop_top_main_cats = shop_main_cats_counts.loc[shop_main_cats_counts.groupby('employee_shop')['count'].idxmax()]

    shop_brands_counts = train_df.groupby(['employee_shop', 'product_brand'])[selection_count_col].sum().reset_index(name='count')
    shop_top_brands = shop_brands_counts.loc[shop_brands_counts.groupby('employee_shop')['count'].idxmax()]
    
    shop_combinations_counts = train_df.groupby('employee_shop').size().reset_index(name='unique_product_combinations_in_shop')

    for _, row in shop_stats_agg.iterrows():
        shop_id = row['employee_shop']
        current_shop_features = {
            'shop_main_category_diversity_selected': int(row['shop_main_category_diversity_selected']),
            'shop_brand_diversity_selected': int(row['shop_brand_diversity_selected']),
            'shop_utility_type_diversity_selected': int(row['shop_utility_type_diversity_selected']),
            'shop_sub_category_diversity_selected': int(row['shop_sub_category_diversity_selected']),
            'shop_most_frequent_main_category_selected': 'NONE',
            'shop_most_frequent_brand_selected': 'NONE',
            'unique_product_combinations_in_shop': 0
        }
        
        top_cat_series = shop_top_main_cats[shop_top_main_cats['employee_shop'] == shop_id]['product_main_category']
        if not top_cat_series.empty:
            current_shop_features['shop_most_frequent_main_category_selected'] = top_cat_series.iloc[0]

        top_brand_series = shop_top_brands[shop_top_brands['employee_shop'] == shop_id]['product_brand']
        if not top_brand_series.empty:
            current_shop_features['shop_most_frequent_brand_selected'] = top_brand_series.iloc[0]
            
        combinations_series = shop_combinations_counts[shop_combinations_counts['employee_shop'] == shop_id]['unique_product_combinations_in_shop']
        if not combinations_series.empty:
            current_shop_features['unique_product_combinations_in_shop'] = int(combinations_series.iloc[0])
            
        historical_features_snapshot[shop_id] = current_shop_features
    
    with open(os.path.join(model_dir, 'historical_features_snapshot.pkl'), 'wb') as f:
        pickle.dump(historical_features_snapshot, f)
    logging.info(f"Saved historical_features_snapshot.pkl for {len(historical_features_snapshot)} shops.")

    # 2. branch_mapping_snapshot
    branch_mapping_snapshot = train_df.groupby('employee_branch')['employee_shop'].unique().apply(list).to_dict()
    with open(os.path.join(model_dir, 'branch_mapping_snapshot.pkl'), 'wb') as f:
        pickle.dump(branch_mapping_snapshot, f)
    logging.info(f"Saved branch_mapping_snapshot.pkl for {len(branch_mapping_snapshot)} branches.")

    # 3. global_defaults_snapshot
    global_defaults_snapshot = {}
    if historical_features_snapshot:
        numeric_cols_for_global = [
            'shop_main_category_diversity_selected', 'shop_brand_diversity_selected',
            'shop_utility_type_diversity_selected', 'shop_sub_category_diversity_selected',
            'unique_product_combinations_in_shop'
        ]
        for col in numeric_cols_for_global:
            values = [data[col] for data in historical_features_snapshot.values() if col in data]
            global_defaults_snapshot[col] = int(np.mean(values)) if values else 5 # Default to 5 if no data

        cat_cols_for_global = ['shop_most_frequent_main_category_selected', 'shop_most_frequent_brand_selected']
        for col in cat_cols_for_global:
            values = [data[col] for data in historical_features_snapshot.values() if col in data]
            global_defaults_snapshot[col] = max(set(values), key=values.count) if values else 'NONE'
    else: # Fallback if historical_features_snapshot is empty
        global_defaults_snapshot = {
            'shop_main_category_diversity_selected': 5, 'shop_brand_diversity_selected': 8,
            'shop_utility_type_diversity_selected': 3, 'shop_sub_category_diversity_selected': 6,
            'unique_product_combinations_in_shop': 45,
            'shop_most_frequent_main_category_selected': 'Home & Kitchen',
            'shop_most_frequent_brand_selected': 'NONE'
        }
    with open(os.path.join(model_dir, 'global_defaults_snapshot.pkl'), 'wb') as f:
        pickle.dump(global_defaults_snapshot, f)
    logging.info(f"Saved global_defaults_snapshot.pkl: {global_defaults_snapshot}")

    # 4. product_relativity_lookup_snapshot.csv
    # This df should be at the granularity of shop, branch, main_category, brand
    product_df = train_df.groupby([
        'employee_shop', 'employee_branch', 'product_main_category', 'product_brand'
    ])[selection_count_col].sum().reset_index()

    if not product_df.empty:
        # Calculate total selections per shop
        total_shop_selections = product_df.groupby('employee_shop')[selection_count_col].sum().rename('total_shop_selections')
        product_df = pd.merge(product_df, total_shop_selections, on='employee_shop', how='left')
        product_df['total_shop_selections'] = product_df['total_shop_selections'].fillna(0)

        # Product Share in Shop (based on main_category and brand combination)
        product_df['product_share_in_shop'] = (product_df[selection_count_col] / product_df['total_shop_selections']).fillna(0)
        
        # Brand Share in Shop
        brand_shop_selections = product_df.groupby(['employee_shop', 'product_brand'])[selection_count_col].sum().reset_index(name='brand_total_selection_in_shop')
        brand_shop_selections = pd.merge(brand_shop_selections, total_shop_selections, on='employee_shop', how='left')
        brand_shop_selections['brand_share_in_shop'] = (brand_shop_selections['brand_total_selection_in_shop'] / brand_shop_selections['total_shop_selections']).fillna(0)
        
        product_df = pd.merge(product_df, brand_shop_selections[['employee_shop', 'product_brand', 'brand_share_in_shop', 'brand_total_selection_in_shop']],
                              on=['employee_shop', 'product_brand'], how='left')

        # Product Rank in Shop (based on selection_count of the main_category/brand combo)
        product_df['product_rank_in_shop'] = product_df.groupby('employee_shop')[selection_count_col].rank(method='dense', ascending=False)

        # Brand Rank in Shop (based on total selection_count of the brand in the shop)
        if 'brand_total_selection_in_shop' in product_df.columns: # Ensure column exists
             product_df['brand_rank_in_shop'] = product_df.groupby('employee_shop')['brand_total_selection_in_shop'].rank(method='dense', ascending=False)
        else: # Fallback if column was not created (e.g. empty brand_shop_selections)
             product_df['brand_rank_in_shop'] = product_df.groupby('employee_shop')[selection_count_col].rank(method='dense', ascending=False)


        # Select and rename columns for the final CSV
        product_relativity_snapshot_df = product_df[[
            'employee_shop', 'employee_branch', 'product_main_category', 'product_brand',
            'product_share_in_shop', 'brand_share_in_shop',
            'product_rank_in_shop', 'brand_rank_in_shop'
        ]].copy()
        
        # Fill NaNs that might have occurred from divisions or merges
        for col in ['product_share_in_shop', 'brand_share_in_shop']:
            product_relativity_snapshot_df[col] = product_relativity_snapshot_df[col].fillna(0.0)
        for col in ['product_rank_in_shop', 'brand_rank_in_shop']:
            product_relativity_snapshot_df[col] = product_relativity_snapshot_df[col].fillna(99.0) # Default high rank

    else:
        logging.warning("product_df for relativity features is empty. Creating an empty snapshot CSV.")
        product_relativity_snapshot_df = pd.DataFrame(columns=[
            'employee_shop', 'employee_branch', 'product_main_category', 'product_brand',
            'product_share_in_shop', 'brand_share_in_shop',
            'product_rank_in_shop', 'brand_rank_in_shop'
        ])

    relativity_csv_path = os.path.join(model_dir, 'product_relativity_features.csv')
    product_relativity_snapshot_df.to_csv(relativity_csv_path, index=False)
    logging.info(f"Saved product_relativity_features.csv with {len(product_relativity_snapshot_df)} rows.")

def train_catboost_model(X_train, y_train, X_val, y_val, exposure_train, exposure_val, cat_features, params, model_name="CatBoost"):
    """Trains a CatBoost model with Poisson objective and returns the trained model and validation metrics."""
    logging.info(f"[{model_name} TRAINING] Training CatBoost Regressor with Poisson loss...")
    logging.info(f"Parameters: {params}")
    
    model = CatBoostRegressor(**params, cat_features=cat_features)
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        sample_weight=exposure_train,
        early_stopping_rounds=params.get('early_stopping_rounds', 50)
    )
    
    y_pred_val = model.predict(X_val)
    # For Poisson, predictions should be non-negative
    y_pred_val = np.maximum(y_pred_val, 0)

    # Calculate metrics appropriate for count data
    from sklearn.metrics import mean_poisson_deviance
    r2_val = r2_score(y_val, y_pred_val, sample_weight=exposure_val)
    mae_val = mean_absolute_error(y_val, y_pred_val, sample_weight=exposure_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val, sample_weight=exposure_val))
    poisson_deviance = mean_poisson_deviance(y_val, y_pred_val, sample_weight=exposure_val)
    
    # Business-weighted MAPE
    weighted_errors = np.abs(y_val - y_pred_val) * exposure_val
    weighted_actuals = y_val * exposure_val
    business_mape = np.mean(weighted_errors / (weighted_actuals + 1e-8)) * 100

    metrics = {
        "r2_validation": r2_val,
        "mae_validation": mae_val,
        "rmse_validation": rmse_val,
        "poisson_deviance": poisson_deviance,
        "business_mape": business_mape
    }
    logging.info(f"[{model_name} VALIDATION] R²: {r2_val:.4f}, MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")
    logging.info(f"[{model_name} VALIDATION] Poisson Deviance: {poisson_deviance:.4f}, Business MAPE: {business_mape:.2f}%")
    return model, metrics

def tune_hyperparameters_optuna(X_train, y_train, X_val, y_val, exposure_train, exposure_val, cat_features, n_trials=300):
    """Tunes CatBoost hyperparameters using Optuna with expanded search space."""
    logging.info("[OPTUNA] Starting Optuna hyperparameter optimization with expanded search space...")
    
    from optuna.pruners import MedianPruner
    from optuna.integration import CatBoostPruningCallback

    def objective(trial):
        # First determine bootstrap type
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli'])
        
        param = {
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 8),  # Shallower for categoricals
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'grow_policy': trial.suggest_categorical('grow_policy', ['Depthwise', 'Lossguide']),
            'bootstrap_type': bootstrap_type,
            'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 50),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            'loss_function': 'Poisson',
            'eval_metric': 'Poisson',
            'random_seed': 42,
            'verbose': 0,
            'early_stopping_rounds': 50
        }
        
        # Add bootstrap-specific parameters
        if bootstrap_type == 'Bayesian':
            param['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 1)
        elif bootstrap_type == 'Bernoulli':
            param['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
        
        # Pruning callback
        pruning_callback = CatBoostPruningCallback(trial, 'Poisson')
        
        model = CatBoostRegressor(**param, cat_features=cat_features)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=exposure_train,
            early_stopping_rounds=param['early_stopping_rounds'],
            callbacks=[pruning_callback],
            verbose=0
        )
        
        # Use validation Poisson score for optimization
        return model.best_score_['validation']['Poisson']

    # Create study with MedianPruner
    study = optuna.create_study(
        direction='minimize',  # Minimize Poisson deviance
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20),
        study_name='catboost_poisson_optimization'
    )
    
    study.optimize(objective, n_trials=n_trials, n_jobs=10, show_progress_bar=True)

    logging.info(f"Optuna: Number of finished trials: {len(study.trials)}")
    best_trial = study.best_trial
    logging.info(f"Optuna: Best trial Poisson score: {best_trial.value:.4f}")
    logging.info(f"Optuna: Best params: {best_trial.params}")
    return best_trial.params

def perform_cross_validation(X, y, y_strata, cat_features, params, n_splits=5):
    """Performs Stratified K-Fold cross-validation."""
    logging.info(f"[CROSS-VALIDATION] Performing Stratified {n_splits}-Fold CV...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    valid_cv_indices = y_strata.dropna().index
    X_for_cv = X.loc[valid_cv_indices]
    y_for_cv = y.loc[valid_cv_indices]
    y_strata_for_cv = y_strata.loc[valid_cv_indices]

    fold_r2_scores = []
    if len(X_for_cv) > 0 and len(y_strata_for_cv) == len(X_for_cv):
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_for_cv, y_strata_for_cv)):
            logging.info(f"  Training fold {fold+1}/{n_splits}...")
            X_train_fold, X_val_fold = X_for_cv.iloc[train_idx], X_for_cv.iloc[val_idx]
            y_train_fold, y_val_fold = y_for_cv.iloc[train_idx], y_for_cv.iloc[val_idx]
            
            cv_model = CatBoostRegressor(**params, cat_features=cat_features)
            cv_model.fit(X_train_fold, y_train_fold,
                         eval_set=[(X_val_fold, y_val_fold)],
                         early_stopping_rounds=params.get('early_stopping_rounds', 50),
                         verbose=0)
            
            y_pred_fold = cv_model.predict(X_val_fold)
            y_pred_fold = np.clip(y_pred_fold, 0, 1)
            fold_r2 = r2_score(y_val_fold, y_pred_fold)
            fold_r2_scores.append(fold_r2)
            logging.info(f"    Fold {fold+1} R²: {fold_r2:.4f}")
        
        logging.info(f"Stratified K-Fold CV R² (mean): {np.mean(fold_r2_scores):.4f}")
        logging.info(f"Stratified K-Fold CV R² (std): {np.std(fold_r2_scores):.4f}")
    else:
        logging.warning("Skipping CV: Not enough valid data after handling NaNs in strata or strata mismatch.")
        return []
    return fold_r2_scores

def analyze_and_save_feature_importance(model, feature_names, save_path):
    """Analyzes feature importance and saves the plot."""
    logging.info("[ANALYSIS] CatBoost Feature Importance Analysis")
    importances = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    logging.info("CatBoost Feature Importance Ranking (Top 20):")
    logging.info(importance_df.head(20).to_string())

    plt.figure(figsize=(12, max(8, len(importance_df.head(20)) * 0.3)))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20), palette='viridis')
    plt.title('CatBoost Feature Importance (Top 20)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Feature importance plot saved to: {save_path}")
    return importance_df

def save_model_artifacts(model, params, metrics, feature_list, cat_feature_list, model_dir, importance_df, numeric_medians):
    """Saves the trained model, parameters, metrics, numeric medians, and other metadata."""
    logging.info(f"[SAVE] Saving model artifacts to {model_dir}...")

    model_path = os.path.join(model_dir, 'catboost_rmse_model.cbm')
    model.save_model(model_path)
    logging.info(f"Trained model saved to: {model_path}")

    params_path = os.path.join(model_dir, 'model_params.pkl')
    with open(params_path, 'wb') as f:
        pickle.dump(params, f)
    logging.info(f"Model parameters saved to: {params_path}")

    metrics_path = os.path.join(model_dir, 'model_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    logging.info(f"Model metrics saved to: {metrics_path}")
    
    metadata = {
        'model_type': 'CatBoost Regressor (Poisson)',
        'catboost_version': catboost.__version__,
        'training_timestamp': datetime.now().isoformat(),
        'features_used': feature_list,
        'categorical_features_in_model': cat_feature_list,
        'numeric_feature_medians': numeric_medians, # Save medians
        'model_parameters': params,
        'performance_metrics': metrics,
        'feature_importance_summary': importance_df.head(20).set_index('feature')['importance'].to_dict()
    }
    metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logging.info(f"Model metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()