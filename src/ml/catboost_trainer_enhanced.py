# src/ml/catboost_trainer_enhanced.py

"""
Enhanced CatBoost Model Training Script with Three-File Architecture

This script implements the optimal data architecture with company-level granularity:
1. present.selection.historic.csv (Training Events)  
2. shop.catalog.csv (Available Gifts)
3. company.employees.csv (Exposure Metadata)

Implements proper exposure calculation, zero-selection records, and eliminates data leakage.
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
from typing import Tuple, Dict, List

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher

import catboost
from catboost import CatBoostRegressor, Pool
import optuna

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "src/data/historical")
MODELS_DIR = os.path.join(BASE_DIR, "models", "catboost_enhanced_model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "catboost_enhanced")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Three-file data paths
SELECTIONS_PATH = os.path.join(DATA_DIR, "present.selection.historic.csv")
CATALOG_PATH = os.path.join(DATA_DIR, "shop.catalog.csv")
COMPANY_EMPLOYEES_PATH = os.path.join(DATA_DIR, "company.employees.csv")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Logging Setup ---
LOG_FILE = os.path.join(LOGS_DIR, f"catboost_enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info(f"CatBoost Enhanced Training - Version: {catboost.__version__}")
logging.info(f"Base directory: {BASE_DIR}")
logging.info(f"Data directory: {DATA_DIR}")
logging.info(f"Models directory: {MODELS_DIR}")

def load_three_file_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the three data files for the optimal architecture.
    
    Returns:
        Tuple of (selections_df, catalog_df, company_employees_df)
    """
    logging.info("[DATA LOAD] Loading three-file architecture data...")
    
    # Load selections data
    try:
        selections_df = pd.read_csv(SELECTIONS_PATH, encoding='utf-8', dtype='str')
        logging.info(f"Loaded selections data: {len(selections_df)} events from {SELECTIONS_PATH}")
        logging.info(f"Selections columns: {list(selections_df.columns)}")
    except FileNotFoundError:
        logging.error(f"ERROR: {SELECTIONS_PATH} not found.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Load catalog data
    try:
        catalog_df = pd.read_csv(CATALOG_PATH, encoding='utf-8', dtype='str')
        logging.info(f"Loaded catalog data: {len(catalog_df)} gift-shop combinations from {CATALOG_PATH}")
        logging.info(f"Catalog columns: {list(catalog_df.columns)}")
    except FileNotFoundError:
        logging.error(f"ERROR: {CATALOG_PATH} not found.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Load company employees data
    try:
        company_employees_df = pd.read_csv(COMPANY_EMPLOYEES_PATH, encoding='utf-8', dtype='str')
        logging.info(f"Loaded company employees data: {len(company_employees_df)} companies from {COMPANY_EMPLOYEES_PATH}")
        logging.info(f"Company employees columns: {list(company_employees_df.columns)}")
    except FileNotFoundError:
        logging.error(f"ERROR: {COMPANY_EMPLOYEES_PATH} not found.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    return selections_df, catalog_df, company_employees_df

def clean_and_prepare_data(selections_df: pd.DataFrame, catalog_df: pd.DataFrame, 
                          company_employees_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans and prepares the three data files.
    """
    logging.info("[DATA CLEAN] Cleaning and preparing data...")
    
    # Clean selections data
    selections_clean = selections_df.copy()
    for col in selections_clean.columns:
        selections_clean[col] = selections_clean[col].astype(str).str.strip('"').str.strip()
    selections_clean = selections_clean.fillna("NONE")
    
    # Normalize categorical columns to lowercase
    categorical_cols_to_lower = [
        'employee_gender', 'product_target_gender',
        'product_utility_type', 'product_durability', 'product_type'
    ]
    for col in categorical_cols_to_lower:
        if col in selections_clean.columns:
            selections_clean[col] = selections_clean[col].str.lower()
    
    # Clean catalog data
    catalog_clean = catalog_df.copy()
    for col in catalog_clean.columns:
        catalog_clean[col] = catalog_clean[col].astype(str).str.strip('"').str.strip()
    catalog_clean = catalog_clean.fillna("NONE")
    
    # Clean company employees data and ensure numeric columns
    company_employees_clean = company_employees_df.copy()
    for col in company_employees_clean.columns:
        company_employees_clean[col] = company_employees_clean[col].astype(str).str.strip('"').str.strip()
    
    # Convert employee counts to numeric
    for col in ['male_count', 'female_count']:
        if col in company_employees_clean.columns:
            company_employees_clean[col] = pd.to_numeric(company_employees_clean[col], errors='coerce').fillna(0)
    
    logging.info(f"Cleaned selections: {len(selections_clean)} events")
    logging.info(f"Cleaned catalog: {len(catalog_clean)} gift-shop combinations")
    logging.info(f"Cleaned company employees: {len(company_employees_clean)} companies")
    
    return selections_clean, catalog_clean, company_employees_clean

def create_training_data_with_exposure(selections_df: pd.DataFrame, catalog_df: pd.DataFrame, 
                                     company_employees_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates training data with proper company-level exposure calculation and zero-selection records.
    
    This implements the optimal architecture from the roadmap that solves exposure and data leakage issues.
    """
    logging.info("[TRAINING DATA] Creating training data with company-level exposure...")
    
    # Define product feature columns
    product_columns = [
        'product_main_category', 'product_sub_category', 'product_brand',
        'product_color', 'product_durability', 'product_target_gender',
        'product_utility_type', 'product_type'
    ]
    
    # Group selections by company, shop, gift, gender and count selections
    selection_groups = selections_df.groupby([
        'shop_id', 'company_cvr', 'gift_id', 'employee_gender'
    ]).agg({
        **{col: 'first' for col in product_columns if col in selections_df.columns}
    }).reset_index()
    
    # Add selection count separately
    selection_counts = selections_df.groupby([
        'shop_id', 'company_cvr', 'gift_id', 'employee_gender'
    ]).size().reset_index(name='selection_count')
    
    # Merge the counts with the features
    selection_groups = pd.merge(
        selection_groups,
        selection_counts,
        on=['shop_id', 'company_cvr', 'gift_id', 'employee_gender'],
        how='left'
    )
    
    logging.info(f"Aggregated selections: {len(selection_groups)} unique combinations")
    
    # Create training records list
    training_data = []
    
    # 1. Add selection records with proper exposure
    for _, row in selection_groups.iterrows():
        shop_id = row['shop_id']
        company_cvr = row['company_cvr'] 
        gift_id = row['gift_id']
        gender = row['employee_gender']
        selection_count = row['selection_count']
        
        # Get exposure from company employees
        company_data = company_employees_df[company_employees_df['company_cvr'] == company_cvr]
        if len(company_data) > 0:
            exposure = company_data[f'{gender}_count'].iloc[0]
            
            # Calculate selection rate
            selection_rate = selection_count / exposure if exposure > 0 else 0.0
            
            # Create training record
            record = {
                'shop_id': shop_id,
                'company_cvr': company_cvr,
                'gift_id': gift_id,
                'employee_gender': gender,
                'selection_count': selection_count,
                'exposure': exposure,
                'selection_rate': selection_rate
            }
            
            # Add product features
            for col in product_columns:
                if col in row:
                    record[col] = row[col]
                else:
                    record[col] = "NONE"
            
            training_data.append(record)
    
    logging.info(f"Added {len(training_data)} selection records with exposure")
    
    # 2. Add zero-selection records for complete universe
    logging.info("[ZERO RECORDS] Adding zero-selection records for unselected gifts...")
    
    # Create set of existing selection keys for fast lookup
    selections_set = set()
    for record in training_data:
        key = (record['shop_id'], record['company_cvr'], record['gift_id'], record['employee_gender'])
        selections_set.add(key)
    
    # Get product features for gifts (we'll need to map gift_id to features)
    # For now, we'll use placeholder features for gifts not in selections
    gift_features_map = {}
    for _, row in selections_df.iterrows():
        gift_id = row['gift_id']
        if gift_id not in gift_features_map:
            features = {}
            for col in product_columns:
                features[col] = row.get(col, "NONE")
            gift_features_map[gift_id] = features
    
    zero_records_added = 0
    # For each company-shop-gift-gender combination not in selections
    for _, company_row in company_employees_df.iterrows():
        company_cvr = company_row['company_cvr']
        
        for _, catalog_row in catalog_df.iterrows():
            shop_id = catalog_row['shop_id']
            gift_id = catalog_row['gift_id']
            
            for gender in ['male', 'female']:
                key = (shop_id, company_cvr, gift_id, gender)
                
                if key not in selections_set:
                    # This is an unselected gift - add zero record
                    exposure = company_row[f'{gender}_count']
                    
                    record = {
                        'shop_id': shop_id,
                        'company_cvr': company_cvr,
                        'gift_id': gift_id,
                        'employee_gender': gender,
                        'selection_count': 0,
                        'exposure': exposure,
                        'selection_rate': 0.0
                    }
                    
                    # Add product features (use from gift features map or defaults)
                    if gift_id in gift_features_map:
                        for col in product_columns:
                            record[col] = gift_features_map[gift_id].get(col, "NONE")
                    else:
                        # Default features for unknown gifts
                        for col in product_columns:
                            record[col] = "NONE"
                    
                    training_data.append(record)
                    zero_records_added += 1
    
    logging.info(f"Added {zero_records_added} zero-selection records")
    
    # Convert to DataFrame
    final_training_df = pd.DataFrame(training_data)
    
    logging.info(f"Final training data shape: {final_training_df.shape}")
    logging.info(f"Selection rate statistics:")
    logging.info(f"  Mean: {final_training_df['selection_rate'].mean():.4f}")
    logging.info(f"  Std: {final_training_df['selection_rate'].std():.4f}")
    logging.info(f"  Min: {final_training_df['selection_rate'].min():.4f}")
    logging.info(f"  Max: {final_training_df['selection_rate'].max():.4f}")
    
    # Validate exposure calculation
    total_selections = final_training_df['selection_count'].sum()
    total_exposure = final_training_df['exposure'].sum()
    overall_rate = total_selections / total_exposure if total_exposure > 0 else 0
    logging.info(f"Overall selection rate: {overall_rate:.4f} ({total_selections} selections / {total_exposure} exposure)")
    
    return final_training_df

def engineer_company_level_features(df: pd.DataFrame, is_training: bool = True, 
                                  lookup_source_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Engineers company-level features that prevent data leakage.
    For training, uses the training data itself. For validation, uses training data as lookup.
    """
    logging.info(f"[FEATURE ENG] Engineering company-level features (training: {is_training})...")
    
    if df.empty:
        logging.warning("Input dataframe is empty, skipping feature engineering")
        return df.copy()
    
    df_featured = df.copy()
    
    # Use lookup source for validation to prevent leakage
    source_df = lookup_source_df if lookup_source_df is not None else df
    
    if source_df.empty:
        logging.warning("Source dataframe is empty, skipping feature engineering")
        return df_featured
    
    # 1. Company-level diversity features (based on what was actually selected)
    company_stats = source_df[source_df['selection_count'] > 0].groupby('company_cvr').agg(
        company_main_category_diversity=('product_main_category', 'nunique'),
        company_brand_diversity=('product_brand', 'nunique'),
        company_utility_type_diversity=('product_utility_type', 'nunique'),
        company_total_selections=('selection_count', 'sum')
    ).reset_index()
    
    # 2. Company-level most frequent features
    # Most frequent main category per company
    company_main_cat_counts = source_df[source_df['selection_count'] > 0].groupby([
        'company_cvr', 'product_main_category'
    ])['selection_count'].sum().reset_index()
    
    if not company_main_cat_counts.empty:
        idx = company_main_cat_counts.groupby('company_cvr')['selection_count'].transform('max') == company_main_cat_counts['selection_count']
        company_top_main_cats = company_main_cat_counts[idx].drop_duplicates(subset=['company_cvr'], keep='first')
        company_top_main_cats = company_top_main_cats[['company_cvr', 'product_main_category']].rename(
            columns={'product_main_category': 'company_most_frequent_main_category'}
        )
        
        # Merge into stats
        company_stats = pd.merge(company_stats, company_top_main_cats, on='company_cvr', how='left')
    
    # Fill missing values with defaults
    numeric_cols = ['company_main_category_diversity', 'company_brand_diversity', 
                   'company_utility_type_diversity', 'company_total_selections']
    for col in numeric_cols:
        if col in company_stats.columns:
            company_stats[col] = company_stats[col].fillna(0)
    
    if 'company_most_frequent_main_category' in company_stats.columns:
        company_stats['company_most_frequent_main_category'] = company_stats['company_most_frequent_main_category'].fillna('NONE')
    else:
        company_stats['company_most_frequent_main_category'] = 'NONE'
    
    # 3. Merge company features into main dataframe
    df_featured = pd.merge(df_featured, company_stats, on='company_cvr', how='left')
    
    # Fill NaNs for companies not seen in training (for validation set)
    for col in numeric_cols:
        if col in df_featured.columns:
            df_featured[col] = df_featured[col].fillna(0)
    
    if 'company_most_frequent_main_category' in df_featured.columns:
        df_featured['company_most_frequent_main_category'] = df_featured['company_most_frequent_main_category'].fillna('NONE')
    
    # 4. Create derived features
    if 'company_most_frequent_main_category' in df_featured.columns:
        df_featured['is_company_most_frequent_main_category'] = (
            df_featured['product_main_category'] == df_featured['company_most_frequent_main_category']
        ).astype(int)
    else:
        df_featured['is_company_most_frequent_main_category'] = 0
    
    logging.info(f"Shape after company-level features: {df_featured.shape}")
    return df_featured

def prepare_features_for_catboost_enhanced(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Prepares features for CatBoost training with selection_rate as target and RMSE loss.
    """
    logging.info("[FEATURES PREP] Preparing features for CatBoost (selection_rate target, RMSE loss)...")
    
    if df.empty or 'selection_rate' not in df.columns:
        logging.error("DataFrame is empty or missing selection_rate column")
        return pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64'), []
    
    # Use selection_rate as target (not selection_count)
    y = pd.to_numeric(df['selection_rate'], errors='coerce').fillna(0)
    
    # Use exposure for scaling predictions (not as offset in RMSE)
    exposure = pd.to_numeric(df['exposure'], errors='coerce').fillna(1)
    
    # Prepare feature matrix
    feature_columns = [
        'shop_id', 'company_cvr', 'gift_id', 'employee_gender',
        'product_main_category', 'product_sub_category', 'product_brand',
        'product_color', 'product_durability', 'product_target_gender',
        'product_utility_type', 'product_type',
        'company_main_category_diversity', 'company_brand_diversity',
        'company_utility_type_diversity', 'company_total_selections',
        'company_most_frequent_main_category', 'is_company_most_frequent_main_category'
    ]
    
    # Select only available columns
    available_columns = [col for col in feature_columns if col in df.columns]
    X = df[available_columns].copy()
    
    # Define categorical features
    categorical_features = [
        'shop_id', 'company_cvr', 'gift_id', 'employee_gender',
        'product_main_category', 'product_sub_category', 'product_brand',
        'product_color', 'product_durability', 'product_target_gender',
        'product_utility_type', 'product_type', 'company_most_frequent_main_category'
    ]
    
    # Keep only categorical features that exist in X
    valid_categorical_features = [col for col in categorical_features if col in X.columns]
    
    # Ensure categorical features are strings
    for col in valid_categorical_features:
        X[col] = X[col].astype(str).fillna("NONE")
    
    # Fill numeric features
    numeric_features = [col for col in X.columns if col not in valid_categorical_features]
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    logging.info(f"Features prepared: X shape {X.shape}, y shape {y.shape}")
    logging.info(f"Categorical features ({len(valid_categorical_features)}): {valid_categorical_features}")
    logging.info(f"Target statistics - mean: {y.mean():.4f}, std: {y.std():.4f}")
    
    return X, y, exposure, valid_categorical_features

def train_catboost_enhanced(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                          categorical_features: List[str], params: Dict) -> Tuple[CatBoostRegressor, Dict]:
    """
    Trains CatBoost model with RMSE loss for selection_rate prediction.
    """
    logging.info("[TRAINING] Training CatBoost with RMSE loss for selection_rate prediction...")
    logging.info(f"Training parameters: {params}")
    
    # Create CatBoost regressor with RMSE loss
    model = CatBoostRegressor(**params, cat_features=categorical_features)
    
    # Create pools (no baseline offset needed for RMSE)
    train_pool = Pool(X_train, label=y_train, cat_features=categorical_features)
    val_pool = Pool(X_val, label=y_val, cat_features=categorical_features)
    
    # Train model
    model.fit(
        train_pool,
        eval_set=[val_pool],
        early_stopping_rounds=params.get('early_stopping_rounds', 50),
        verbose=100
    )
    
    # Make predictions
    y_pred_val = model.predict(X_val)
    
    # Calculate metrics
    r2_val = r2_score(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    
    # Calculate business metrics
    business_mape = np.mean(np.abs(y_val - y_pred_val) / (y_val + 1e-8)) * 100
    
    metrics = {
        "r2_validation": r2_val,
        "mae_validation": mae_val,
        "rmse_validation": rmse_val,
        "business_mape": business_mape
    }
    
    logging.info(f"[VALIDATION] R²: {r2_val:.4f}, MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")
    logging.info(f"[VALIDATION] Business MAPE: {business_mape:.2f}%")
    
    return model, metrics

def save_enhanced_model_artifacts(model: CatBoostRegressor, params: Dict, metrics: Dict, 
                                feature_list: List[str], categorical_features: List[str]):
    """
    Saves the enhanced model artifacts.
    """
    logging.info(f"[SAVE] Saving enhanced model artifacts to {MODELS_DIR}...")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'catboost_enhanced_model.cbm')
    model.save_model(model_path)
    logging.info(f"Model saved to: {model_path}")
    
    # Save parameters
    with open(os.path.join(MODELS_DIR, 'model_params.pkl'), 'wb') as f:
        pickle.dump(params, f)
    
    # Save metrics
    with open(os.path.join(MODELS_DIR, 'model_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    # Save metadata
    metadata = {
        'model_type': 'CatBoost Regressor (Enhanced - RMSE)',
        'target': 'selection_rate',
        'loss_function': 'RMSE',
        'catboost_version': catboost.__version__,
        'training_timestamp': datetime.now().isoformat(),
        'features_used': feature_list,
        'categorical_features': categorical_features,
        'architecture': 'Three-file company-level architecture',
        'performance_metrics': metrics
    }
    
    with open(os.path.join(MODELS_DIR, 'model_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    logging.info("All enhanced model artifacts saved successfully")

def main():
    """
    Main training pipeline with enhanced three-file architecture.
    """
    logging.info("--- Starting Enhanced CatBoost Training Pipeline ---")
    logging.info("Implementing optimal three-file architecture with company-level granularity")
    
    # 1. Load three-file data
    selections_df, catalog_df, company_employees_df = load_three_file_data()
    
    if selections_df.empty or catalog_df.empty or company_employees_df.empty:
        logging.error("Failed to load required data files. Exiting.")
        return
    
    # 2. Clean and prepare data
    selections_clean, catalog_clean, company_employees_clean = clean_and_prepare_data(
        selections_df, catalog_df, company_employees_df
    )
    
    # 3. Create training data with proper exposure
    training_df = create_training_data_with_exposure(
        selections_clean, catalog_clean, company_employees_clean
    )
    
    if training_df.empty:
        logging.error("Failed to create training data. Exiting.")
        return
    
    # 4. Split data by companies to prevent leakage
    logging.info("[SPLIT] Creating company-based train/test split...")
    
    unique_companies = training_df['company_cvr'].unique()
    train_companies, val_companies = train_test_split(
        unique_companies, test_size=0.2, random_state=42
    )
    
    train_df = training_df[training_df['company_cvr'].isin(train_companies)].copy()
    val_df = training_df[training_df['company_cvr'].isin(val_companies)].copy()
    
    logging.info(f"Train companies: {len(train_companies)}, Validation companies: {len(val_companies)}")
    logging.info(f"Train records: {len(train_df)}, Validation records: {len(val_df)}")
    
    # Verify no company leakage
    train_companies_set = set(train_df['company_cvr'].unique())
    val_companies_set = set(val_df['company_cvr'].unique())
    company_overlap = train_companies_set & val_companies_set
    
    if company_overlap:
        logging.error(f"CRITICAL: Company leakage detected! {len(company_overlap)} companies in both sets")
        raise ValueError("Company leakage detected in data split")
    else:
        logging.info("✅ No company leakage detected")
    
    # 5. Engineer features
    train_featured = engineer_company_level_features(train_df, is_training=True)
    val_featured = engineer_company_level_features(val_df, is_training=False, lookup_source_df=train_df)
    
    # 6. Prepare features for CatBoost
    X_train, y_train, exposure_train, cat_features = prepare_features_for_catboost_enhanced(train_featured)
    X_val, y_val, exposure_val, _ = prepare_features_for_catboost_enhanced(val_featured)
    
    # Ensure validation features match training features
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    
    # Fill categorical columns with "NONE" instead of 0
    for col in cat_features:
        if col in X_val.columns:
            X_val[col] = X_val[col].replace(0, "NONE").astype(str)
    
    if X_train.empty or y_train.empty:
        logging.error("Feature preparation failed. Exiting.")
        return
    
    # 7. Train model
    model_params = {
        'iterations': 1000,
        'loss_function': 'RMSE',  # Using RMSE for rate prediction
        'eval_metric': 'RMSE',
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'early_stopping_rounds': 50,
        'verbose': 100
    }
    
    trained_model, final_metrics = train_catboost_enhanced(
        X_train, y_train, X_val, y_val, cat_features, model_params
    )
    
    # 8. Save model artifacts
    save_enhanced_model_artifacts(
        trained_model, model_params, final_metrics, 
        list(X_train.columns), cat_features
    )
    
    logging.info("--- Enhanced CatBoost Training Pipeline Completed Successfully ---")
    logging.info("Model trained with optimal three-file architecture and company-level granularity")

class EnhancedCatBoostTrainer:
    """
    Enhanced CatBoost Trainer with Three-File Architecture
    
    Provides a class interface for the enhanced training pipeline that implements
    company-level granularity and solves exposure/data leakage issues.
    """
    
    def __init__(self):
        """Initialize the enhanced trainer."""
        self.models_dir = MODELS_DIR
        self.data_dir = DATA_DIR
        logging.info("Enhanced CatBoost Trainer initialized")
    
    def create_training_data_with_exposure(self, selections_df: pd.DataFrame, catalog_df: pd.DataFrame,
                                         company_employees_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates training data with proper company-level exposure calculation.
        
        Args:
            selections_df: Selection events data
            catalog_df: Shop catalog data
            company_employees_df: Company employee counts
            
        Returns:
            Training dataframe with exposure and selection rates
        """
        # Clean the data first
        selections_clean, catalog_clean, company_employees_clean = clean_and_prepare_data(
            selections_df, catalog_df, company_employees_df
        )
        
        # Create training data with exposure
        return create_training_data_with_exposure(
            selections_clean, catalog_clean, company_employees_clean
        )
    
    def train_model(self, training_data: pd.DataFrame) -> Dict:
        """
        Trains the CatBoost model with the enhanced pipeline.
        
        Args:
            training_data: Training dataframe with exposure and selection rates
            
        Returns:
            Dictionary with model info and performance metrics
        """
        if training_data.empty:
            raise ValueError("Training data is empty")
        
        logging.info("[CLASS] Starting enhanced model training...")
        
        # Split data by companies to prevent leakage
        unique_companies = training_data['company_cvr'].unique()
        train_companies, val_companies = train_test_split(
            unique_companies, test_size=0.2, random_state=42
        )
        
        train_df = training_data[training_data['company_cvr'].isin(train_companies)].copy()
        val_df = training_data[training_data['company_cvr'].isin(val_companies)].copy()
        
        logging.info(f"Train companies: {len(train_companies)}, Validation companies: {len(val_companies)}")
        
        # Verify no company leakage
        train_companies_set = set(train_df['company_cvr'].unique())
        val_companies_set = set(val_df['company_cvr'].unique())
        company_overlap = train_companies_set & val_companies_set
        
        if company_overlap:
            raise ValueError("Company leakage detected in data split")
        
        # Engineer features
        train_featured = engineer_company_level_features(train_df, is_training=True)
        val_featured = engineer_company_level_features(val_df, is_training=False, lookup_source_df=train_df)
        
        # Prepare features for CatBoost
        X_train, y_train, exposure_train, cat_features = prepare_features_for_catboost_enhanced(train_featured)
        X_val, y_val, exposure_val, _ = prepare_features_for_catboost_enhanced(val_featured)
        
        # Ensure validation features match training features
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
        
        # Fill categorical columns with "NONE" instead of 0
        for col in cat_features:
            if col in X_val.columns:
                X_val[col] = X_val[col].replace(0, "NONE").astype(str)
        
        # Model parameters
        model_params = {
            'iterations': 1000,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'early_stopping_rounds': 50,
            'verbose': 100
        }
        
        # Train model
        trained_model, final_metrics = train_catboost_enhanced(
            X_train, y_train, X_val, y_val, cat_features, model_params
        )
        
        # Save model artifacts
        save_enhanced_model_artifacts(
            trained_model, model_params, final_metrics,
            list(X_train.columns), cat_features
        )
        
        # Return model info
        model_path = os.path.join(MODELS_DIR, 'catboost_enhanced_model.cbm')
        
        # Perform cross-validation for more robust metrics
        cv_scores = []
        cv_folds = 3
        
        # Simple CV with company-based splits
        companies_array = unique_companies
        np.random.seed(42)
        np.random.shuffle(companies_array)
        
        fold_size = len(companies_array) // cv_folds
        
        for fold in range(cv_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < cv_folds - 1 else len(companies_array)
            
            val_companies_cv = companies_array[start_idx:end_idx]
            train_companies_cv = np.concatenate([
                companies_array[:start_idx],
                companies_array[end_idx:]
            ])
            
            train_df_cv = training_data[training_data['company_cvr'].isin(train_companies_cv)].copy()
            val_df_cv = training_data[training_data['company_cvr'].isin(val_companies_cv)].copy()
            
            if len(train_df_cv) > 0 and len(val_df_cv) > 0:
                # Quick feature prep for CV
                train_featured_cv = engineer_company_level_features(train_df_cv, is_training=True)
                val_featured_cv = engineer_company_level_features(val_df_cv, is_training=False, lookup_source_df=train_df_cv)
                
                X_train_cv, y_train_cv, _, cat_features_cv = prepare_features_for_catboost_enhanced(train_featured_cv)
                X_val_cv, y_val_cv, _, _ = prepare_features_for_catboost_enhanced(val_featured_cv)
                
                X_val_cv = X_val_cv.reindex(columns=X_train_cv.columns, fill_value=0)
                for col in cat_features_cv:
                    if col in X_val_cv.columns:
                        X_val_cv[col] = X_val_cv[col].replace(0, "NONE").astype(str)
                
                # Quick model for CV
                cv_params = model_params.copy()
                cv_params['iterations'] = 300  # Faster for CV
                cv_params['verbose'] = False
                
                cv_model = CatBoostRegressor(**cv_params, cat_features=cat_features_cv)
                cv_model.fit(X_train_cv, y_train_cv)
                
                y_pred_cv = cv_model.predict(X_val_cv)
                cv_r2 = r2_score(y_val_cv, y_pred_cv)
                cv_scores.append(cv_r2)
        
        # Calculate CV metrics
        cv_r2_mean = np.mean(cv_scores) if cv_scores else final_metrics.get('r2_validation', 0)
        cv_r2_std = np.std(cv_scores) if cv_scores else 0
        
        model_info = {
            'model_path': model_path,
            'model_type': 'CatBoost Regressor (Enhanced - RMSE)',
            'target': 'selection_rate',
            'architecture': 'Three-file company-level architecture',
            'performance_metrics': {
                **final_metrics,
                'cv_r2_mean': cv_r2_mean,
                'cv_r2_std': cv_r2_std,
                'cv_folds': len(cv_scores)
            },
            'training_records': len(training_data),
            'training_companies': len(unique_companies),
            'feature_count': len(X_train.columns),
            'categorical_features': len(cat_features)
        }
        
        logging.info(f"[CLASS] Training completed. CV R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
        return model_info


if __name__ == "__main__":
    main()