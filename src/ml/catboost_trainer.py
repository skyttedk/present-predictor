# src/ml/catboost_trainer.py

"""
CatBoost Model Training Script

This script implements the CatBoost Regressor model training pipeline,
migrated from the 'notebooks/catboost_implementation.ipynb' notebook.
It includes data loading, preprocessing, feature engineering,
CatBoost model training with Poisson loss, hyperparameter tuning with Optuna,
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
MODELS_DIR = os.path.join(BASE_DIR, "models", "catboost_poisson_model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "catboost_poisson")
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
    logging.info(f"Data cleaning complete: {len(cleaned_data)} records")
    return cleaned_data

def aggregate_data(cleaned_data: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Aggregates data to unique product-employee combinations."""
    if cleaned_data.empty:
        logging.warning("Cleaned data is empty, skipping aggregation.")
        return pd.DataFrame(), []

    grouping_cols = [
        'employee_shop', 'employee_branch', 'employee_gender',
        'product_main_category', 'product_sub_category', 'product_brand',
        'product_color', 'product_durability', 'product_target_gender',
        'product_utility_type', 'product_type'
    ]
    logging.info(f"[AGGREGATE] Grouping by {len(grouping_cols)} features: {grouping_cols}")
    agg_data = cleaned_data.groupby(grouping_cols).size().reset_index(name='selection_count')

    if not agg_data.empty:
        compression_ratio = len(cleaned_data) / len(agg_data) if len(agg_data) > 0 else 0
        logging.info("Aggregation complete:")
        logging.info(f"  {len(cleaned_data)} events -> {len(agg_data)} unique combinations")
        logging.info(f"  Compression ratio: {compression_ratio:.1f}x")
    else:
        logging.warning("agg_data is empty after grouping.")
    return agg_data, grouping_cols

def engineer_shop_assortment_features(agg_data: pd.DataFrame, selection_count_col: str = 'selection_count') -> pd.DataFrame:
    """Reuses the non-leaky shop assortment features from breakthrough_training.ipynb."""
    logging.info("[FEATURE ENG] Engineering existing shop assortment features...")
    if agg_data.empty:
        logging.warning("Skipping existing shop features as agg_data is empty.")
        return agg_data.copy()

    # Ensure selection_count_col is numeric for aggregations
    agg_data[selection_count_col] = pd.to_numeric(agg_data[selection_count_col], errors='coerce').fillna(0)

    shop_summary = agg_data.groupby('employee_shop').agg(
        unique_product_combinations_in_shop=('product_main_category', 'count'), # Count of rows per shop
        distinct_main_categories_in_shop=('product_main_category', 'nunique'),
        distinct_sub_categories_in_shop=('product_sub_category', 'nunique'),
        distinct_brands_in_shop=('product_brand', 'nunique'),
        distinct_utility_types_in_shop=('product_utility_type', 'nunique')
    ).reset_index()

    shop_main_cat_counts = agg_data.groupby(['employee_shop', 'product_main_category'])[selection_count_col].sum().reset_index()
    idx = shop_main_cat_counts.groupby(['employee_shop'])[selection_count_col].transform(max) == shop_main_cat_counts[selection_count_col]
    shop_top_main_cats = shop_main_cat_counts[idx].drop_duplicates(subset=['employee_shop'], keep='first')
    shop_top_main_cats = shop_top_main_cats[['employee_shop', 'product_main_category']].rename(
        columns={'product_main_category': 'shop_most_frequent_main_category_selected'}
    )

    shop_brand_counts = agg_data.groupby(['employee_shop', 'product_brand'])[selection_count_col].sum().reset_index()
    idx_brand = shop_brand_counts.groupby(['employee_shop'])[selection_count_col].transform(max) == shop_brand_counts[selection_count_col]
    shop_top_brands = shop_brand_counts[idx_brand].drop_duplicates(subset=['employee_shop'], keep='first')
    shop_top_brands = shop_top_brands[['employee_shop', 'product_brand']].rename(
        columns={'product_brand': 'shop_most_frequent_brand_selected'}
    )

    shop_features_df = shop_summary.copy()
    if not shop_top_main_cats.empty:
        shop_features_df = pd.merge(shop_features_df, shop_top_main_cats, on='employee_shop', how='left')
    if not shop_top_brands.empty:
        shop_features_df = pd.merge(shop_features_df, shop_top_brands, on='employee_shop', how='left')

    shop_features_df = shop_features_df.rename(columns={
        'distinct_main_categories_in_shop': 'shop_main_category_diversity_selected',
        'distinct_brands_in_shop': 'shop_brand_diversity_selected',
        'distinct_utility_types_in_shop': 'shop_utility_type_diversity_selected',
        'distinct_sub_categories_in_shop': 'shop_sub_category_diversity_selected'
    })

    agg_data_with_shop_features = pd.merge(agg_data, shop_features_df, on='employee_shop', how='left')

    if 'unique_product_combinations_in_shop' in agg_data_with_shop_features.columns:
        agg_data_with_shop_features['unique_product_combinations_in_shop'] = pd.to_numeric(
            agg_data_with_shop_features['unique_product_combinations_in_shop'], errors='coerce'
        ).fillna(0)

    # Product-Relative-to-Shop Features
    df_temp = agg_data_with_shop_features.copy()
    if 'shop_most_frequent_main_category_selected' in df_temp.columns:
        df_temp['is_shop_most_frequent_main_category'] = (
            df_temp['product_main_category'] == df_temp['shop_most_frequent_main_category_selected']
        ).astype(int)
    else:
        df_temp['is_shop_most_frequent_main_category'] = 0

    if 'shop_most_frequent_brand_selected' in df_temp.columns:
        df_temp['is_shop_most_frequent_brand'] = (
            df_temp['product_brand'] == df_temp['shop_most_frequent_brand_selected']
        ).astype(int)
    else:
        df_temp['is_shop_most_frequent_brand'] = 0
    
    logging.info(f"Shape after existing shop features: {df_temp.shape}")
    return df_temp

def engineer_new_interaction_features(df: pd.DataFrame, n_interaction_features: int = 10) -> pd.DataFrame:
    """Engineers new interaction features using FeatureHasher."""
    logging.info(f"[FEATURE ENG] Engineering new interaction features (target: {n_interaction_features}). Initial shape: {df.shape}")
    if df.empty:
        logging.warning("Skipping new interaction feature engineering as input df is empty.")
        return df.copy()

    hasher = FeatureHasher(n_features=n_interaction_features, input_type='string')
    
    # Ensure required columns exist, handle if not
    if 'employee_branch' not in df.columns or 'product_main_category' not in df.columns:
        logging.warning("Required columns for interaction hashing ('employee_branch', 'product_main_category') not found. Skipping interaction features.")
        return df

    interaction_strings = df.apply(
        lambda x: f"{x.get('employee_branch', 'NONE')}_{x.get('product_main_category', 'NONE')}", axis=1
    )
    interaction_hashed_features = hasher.transform(interaction_strings.astype(str).apply(lambda s: [s])).toarray()

    for i in range(interaction_hashed_features.shape[1]):
        df[f'interaction_hash_{i}'] = interaction_hashed_features[:, i]
    
    logging.info(f"Shape after new interaction features: {df.shape}")
    return df


def engineer_product_relativity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers features describing a product's historical performance relative to its shop."""
    logging.info("[FEATURE ENG] Engineering product relativity features...")
    if df.empty:
        logging.warning("Skipping product relativity features as df is empty.")
        return df.copy()

    # Ensure selection_count is numeric
    df['selection_count'] = pd.to_numeric(df['selection_count'], errors='coerce').fillna(0)

    # --- Calculate total selections for normalization ---
    # Total selections per shop
    shop_total_selections = df.groupby('employee_shop')['selection_count'].transform('sum')
    
    # Total selections per brand within each shop
    brand_total_selections_in_shop = df.groupby(['employee_shop', 'product_brand'])['selection_count'].transform('sum')

    # --- Engineer Share Features ---
    # Use a small epsilon to avoid division by zero
    epsilon = 1e-9
    df['product_share_in_shop'] = df['selection_count'] / (shop_total_selections + epsilon)
    df['brand_share_in_shop'] = brand_total_selections_in_shop / (shop_total_selections + epsilon)

    # --- Engineer Rank Features ---
    df['product_rank_in_shop'] = df.groupby('employee_shop')['selection_count'].rank(method='dense', ascending=False)
    
    # For brand rank, we need to rank the total selections for the brand, not the row-level count
    # First, let's add the brand total selections as a temporary column
    df['temp_brand_total_selections'] = brand_total_selections_in_shop
    df['brand_rank_in_shop'] = df.groupby('employee_shop')['temp_brand_total_selections'].rank(method='dense', ascending=False)
    df = df.drop(columns=['temp_brand_total_selections'])

    logging.info(f"Shape after product relativity features: {df.shape}")
    logging.info(f"New features created: product_share_in_shop, brand_share_in_shop, product_rank_in_shop, brand_rank_in_shop")
    
    return df


def prepare_features_for_catboost(final_features_df: pd.DataFrame, grouping_cols: list) -> tuple[pd.DataFrame, pd.Series, pd.Series, list, dict]:
    """Prepares X, y, y_strata, identifies categorical features, and calculates numeric medians for CatBoost."""
    logging.info("[FEATURES PREP] Preparing final X, y for CatBoost and calculating medians...")
    if final_features_df.empty or 'selection_count' not in final_features_df.columns:
        logging.warning("Skipping final feature preparation as final_features_df is empty or 'selection_count' is missing.")
        return pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64'), [], {}

    y = pd.to_numeric(final_features_df['selection_count'], errors='coerce').fillna(0)
    X = final_features_df.drop(columns=['selection_count']).copy()

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
                numeric_medians[col] = X[col].median()
            else:
                # If still not numeric, it might be problematic or an unexpected type.
                # For now, we'll assign a default median of 0 for such cases,
                # but this should be reviewed if it occurs frequently.
                numeric_medians[col] = 0.0
                logging.warning(f"Column '{col}' is not numeric after conversion attempts. Default median set to 0.")

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

    y_strata = pd.cut(y, bins=[-1, 0, 1, 2, 5, 10, np.inf], labels=[0, 1, 2, 3, 4, 5], include_lowest=True)

    logging.info(f"Final X features shape: {X.shape}")
    logging.info(f"Target y shape: {y.shape}")
    logging.info(f"Categorical features for CatBoost ({len(valid_categorical_features)}): {valid_categorical_features}")
    logging.info(f"Categorical feature indices: {final_cat_feature_indices}")
    logging.info(f"Calculated numeric medians: {numeric_medians}")
    if not y_strata.empty:
      logging.info(f"Stratification distribution:\n{y_strata.value_counts().sort_index().to_dict()}")
    
    return X, y, y_strata, valid_categorical_features, numeric_medians


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

    # 3. Engineer Shop Assortment Features
    features_with_shop = engineer_shop_assortment_features(agg_df)

    # 4. Engineer New Interaction Features
    features_with_interactions = engineer_new_interaction_features(features_with_shop)

    # 5. Engineer Product Relativity Features
    final_features_df = engineer_product_relativity_features(features_with_interactions)

    # Save the lookup table for the predictor
    product_relativity_features_path = os.path.join(MODELS_DIR, 'product_relativity_features.csv')
    lookup_cols = base_grouping_cols + ['product_share_in_shop', 'brand_share_in_shop', 'product_rank_in_shop', 'brand_rank_in_shop']
    # Ensure columns exist before trying to save them
    lookup_cols_exist = [col for col in lookup_cols if col in final_features_df.columns]
    if lookup_cols_exist:
        final_features_df[lookup_cols_exist].to_csv(product_relativity_features_path, index=False)
        logging.info(f"Product relativity feature lookup table saved to {product_relativity_features_path}")
    else:
        logging.warning("Could not save product relativity lookup table as no lookup columns were found.")


    # 6. Prepare Features for CatBoost
    X, y, y_strata, cat_features_for_model, numeric_medians = prepare_features_for_catboost(final_features_df, base_grouping_cols)
    
    if X.empty or y.empty:
        logging.error("Feature preparation failed. Exiting.")
        return

    logging.info("--- Data Preparation Complete ---")
    logging.info(f"X shape: {X.shape}, y shape: {y.shape}, Num Cat Features: {len(cat_features_for_model)}")
    logging.info(f"Numeric Medians: {numeric_medians}")
    
    # 6. Train Initial CatBoost Model
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_strata
    )
    
    initial_model_params = {
        'iterations': 1000,
        'loss_function': 'Poisson',
        'eval_metric': 'R2',
        'random_seed': 42,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'early_stopping_rounds': 50,
        'verbose': 100
    }
    trained_initial_model, initial_metrics = train_catboost_model(
        X_train, y_train, X_val, y_val, cat_features_for_model, initial_model_params, "Initial"
    )

    if not trained_initial_model:
        logging.error("Initial model training failed. Exiting.")
        return

    # 7. Hyperparameter Tuning with Optuna
    best_params_optuna = tune_hyperparameters_optuna(
        X_train, y_train, X_val, y_val, cat_features_for_model, n_trials=15
    )
    
    if not best_params_optuna:
        logging.warning("Optuna tuning did not find best parameters, using initial model parameters for further steps.")
        best_params_final = initial_model_params
    else:
        best_params_final = best_params_optuna
        best_params_final['loss_function'] = 'Poisson'
        best_params_final['eval_metric'] = 'R2'
        best_params_final['random_seed'] = 42
        best_params_final['verbose'] = 100
        if 'early_stopping_rounds' not in best_params_final:
             best_params_final['early_stopping_rounds'] = initial_model_params.get('early_stopping_rounds', 50)

    # 8. Train Final Model with Best/Chosen Parameters
    trained_final_model, final_metrics_val = train_catboost_model(
        X_train, y_train, X_val, y_val, cat_features_for_model, best_params_final, "Final Tuned"
    )

    if not trained_final_model:
        logging.error("Final model training failed. Exiting.")
        return

    # 9. Cross-Validation with Final Parameters
    cv_scores = perform_cross_validation(
        X, y, y_strata, cat_features_for_model, best_params_final, n_splits=5
    )

    # 10. Feature Importance
    importance_df = analyze_and_save_feature_importance(
        trained_final_model, X.columns, os.path.join(REPORTS_DIR, "figures", "catboost_feature_importance.png")
    )

    # 11. Save Artifacts
    save_model_artifacts(
        trained_final_model,
        best_params_final,
        {**final_metrics_val, "cv_r2_mean": np.mean(cv_scores) if cv_scores else None, "cv_r2_std": np.std(cv_scores) if cv_scores else None},
        list(X.columns),
        cat_features_for_model,
        MODELS_DIR,
        importance_df,
        numeric_medians # Pass medians to save_model_artifacts
    )

    logging.info("--- CatBoost Model Training Pipeline Finished ---")

def train_catboost_model(X_train, y_train, X_val, y_val, cat_features, params, model_name="CatBoost"):
    """Trains a CatBoost model and returns the trained model and validation metrics."""
    logging.info(f"[{model_name} TRAINING] Training CatBoost Regressor...")
    logging.info(f"Parameters: {params}")
    
    model = CatBoostRegressor(**params, cat_features=cat_features)
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=params.get('early_stopping_rounds', 50)
    )
    
    y_pred_val = model.predict(X_val)
    y_pred_val = np.maximum(0, y_pred_val)

    r2_val = r2_score(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

    metrics = {
        "r2_validation": r2_val,
        "mae_validation": mae_val,
        "rmse_validation": rmse_val
    }
    logging.info(f"[{model_name} VALIDATION] R²: {r2_val:.4f}, MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")
    return model, metrics

def tune_hyperparameters_optuna(X_train, y_train, X_val, y_val, cat_features, n_trials=15):
    """Tunes CatBoost hyperparameters using Optuna."""
    logging.info("[OPTUNA] Starting Optuna hyperparameter optimization...")

    def objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 400, 1500, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 20.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'loss_function': 'Poisson',
            'eval_metric': 'R2',
            'random_seed': 42,
            'verbose': 0,
            'early_stopping_rounds': 50
        }
        model = CatBoostRegressor(**param, cat_features=cat_features)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=param['early_stopping_rounds'], verbose=0)
        preds = model.predict(X_val)
        preds = np.maximum(0, preds)
        r2 = r2_score(y_val, preds)
        return r2

    study = optuna.create_study(direction='maximize', study_name='catboost_poisson_r2_script')
    study.optimize(objective, n_trials=n_trials)

    logging.info(f"Optuna: Number of finished trials: {len(study.trials)}")
    best_trial = study.best_trial
    logging.info(f"Optuna: Best trial R2: {best_trial.value:.4f}")
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
            y_pred_fold = np.maximum(0, y_pred_fold)
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

    model_path = os.path.join(model_dir, 'catboost_poisson_model.cbm')
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