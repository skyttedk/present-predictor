"""
Data preprocessing pipeline for the Predictive Gift Selection System.
Handles historical data loading, cleaning, and aggregation for ML training.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from pathlib import Path

from .schemas.data_models import HistoricalRecord, load_historical_data, create_dataframe_from_records
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing pipeline for historical gift selection data.
    Aggregates selection events and prepares training data for XGBoost.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.settings = get_settings()
        self.raw_data: Optional[pd.DataFrame] = None
        self.aggregated_data: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        
    def load_historical_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical data from CSV file.
        
        Args:
            file_path: Path to historical data file. If None, uses default path.
            
        Returns:
            Raw historical data as DataFrame
        """
        if file_path is None:
            file_path = "src/data/historical/present.selection.historic.csv"
        
        try:
            # Try different encodings to handle various file formats
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            df = None
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"Could not read CSV file with any of the tried encodings: {encodings_to_try}")
            
            # Clean the data
            df = self._clean_raw_data(df)
            
            # Store raw data
            self.raw_data = df
            
            logger.info(f"Loaded {len(df)} historical records from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load historical data from {file_path}: {e}")
            raise
    
    def _clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw historical data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove quotes from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip('"').str.strip()
        
        # Handle missing values
        df = df.fillna("NONE")
        
        # Standardize categorical values
        df = self._standardize_categories(df)
        
        return df
    
    def _standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical values for consistency."""
        df = df.copy()
        
        # Standardize gender values
        if 'employee_gender' in df.columns:
            df['employee_gender'] = df['employee_gender'].str.lower()
            
        if 'product_target_gender' in df.columns:
            df['product_target_gender'] = df['product_target_gender'].str.lower()
        
        # Standardize utility types
        if 'product_utility_type' in df.columns:
            df['product_utility_type'] = df['product_utility_type'].str.lower()
        
        # Standardize durability
        if 'product_durability' in df.columns:
            df['product_durability'] = df['product_durability'].str.lower()
        
        # Standardize usage type
        if 'product_type' in df.columns:
            df['product_type'] = df['product_type'].str.lower()
        
        return df
    
    def aggregate_selection_events(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Aggregate selection events by counting occurrences.
        Each row in historical data represents a selection event.
        
        Args:
            df: DataFrame to aggregate. If None, uses loaded data.
            
        Returns:
            Aggregated DataFrame with selection counts
        """
        if df is None:
            if self.raw_data is None:
                raise ValueError("No data loaded. Call load_historical_data() first.")
            df = self.raw_data
        
        # Define grouping columns (all categorical features)
        grouping_columns = [
            'employee_shop',
            'employee_branch', 
            'employee_gender',
            'product_main_category',
            'product_sub_category',
            'product_brand',
            'product_color',
            'product_durability',
            'product_target_gender',
            'product_utility_type',
            'product_type'
        ]
        
        # Ensure all grouping columns exist
        missing_columns = [col for col in grouping_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")
        
        # Aggregate by counting selection events
        aggregated = df.groupby(grouping_columns).size().reset_index(name='selection_count')
        
        # Store feature columns (excluding target)
        self.feature_columns = grouping_columns
        self.aggregated_data = aggregated
        
        logger.info(f"Aggregated {len(df)} events into {len(aggregated)} unique combinations")
        return aggregated
    
    def create_training_features(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create feature matrix and target variable for ML training.
        
        Args:
            df: Aggregated DataFrame. If None, uses stored aggregated data.
            
        Returns:
            Tuple of (features_df, target_series)
        """
        if df is None:
            if self.aggregated_data is None:
                raise ValueError("No aggregated data. Call aggregate_selection_events() first.")
            df = self.aggregated_data
        
        # Separate features and target
        features_df = df[self.feature_columns].copy()
        target_series = df['selection_count']
        
        # Encode categorical features
        features_encoded = self._encode_categorical_features(features_df)
        
        logger.info(f"Created training features: {features_encoded.shape}")
        logger.info(f"Target distribution - Mean: {target_series.mean():.2f}, Max: {target_series.max()}")
        
        return features_encoded, target_series
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for ML training.
        Uses label encoding for now, can be enhanced with one-hot encoding.
        
        Args:
            df: Features DataFrame
            
        Returns:
            Encoded features DataFrame
        """
        from sklearn.preprocessing import LabelEncoder
        
        encoded_df = df.copy()
        self.label_encoders = {}
        
        for column in df.columns:
            if df[column].dtype == 'object':
                le = LabelEncoder()
                encoded_df[column] = le.fit_transform(df[column].astype(str))
                self.label_encoders[column] = le
        
        return encoded_df
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the processed data."""
        summary = {}
        
        if self.raw_data is not None:
            summary['raw_data'] = {
                'total_records': len(self.raw_data),
                'unique_employees': self.raw_data['employee_gender'].value_counts().to_dict(),
                'unique_categories': self.raw_data['product_main_category'].nunique(),
                'unique_brands': self.raw_data['product_brand'].nunique(),
            }
        
        if self.aggregated_data is not None:
            summary['aggregated_data'] = {
                'unique_combinations': len(self.aggregated_data),
                'total_selections': self.aggregated_data['selection_count'].sum(),
                'avg_selections_per_combination': self.aggregated_data['selection_count'].mean(),
                'max_selections': self.aggregated_data['selection_count'].max(),
                'selection_distribution': self.aggregated_data['selection_count'].value_counts().head(10).to_dict()
            }
        
        return summary
    
    def get_category_insights(self) -> Dict[str, Any]:
        """Get insights about product categories and employee preferences."""
        if self.aggregated_data is None:
            return {}
        
        insights = {}
        
        # Most popular product categories
        category_popularity = (
            self.aggregated_data.groupby('product_main_category')['selection_count']
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        )
        insights['popular_categories'] = category_popularity
        
        # Gender preferences
        gender_preferences = (
            self.aggregated_data.groupby(['employee_gender', 'product_main_category'])['selection_count']
            .sum()
            .unstack(fill_value=0)
            .to_dict()
        )
        insights['gender_preferences'] = gender_preferences
        
        # Brand popularity
        brand_popularity = (
            self.aggregated_data.groupby('product_brand')['selection_count']
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        )
        insights['popular_brands'] = brand_popularity
        
        return insights


def load_and_preprocess_historical_data(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, DataPreprocessor]:
    """
    Convenience function to load and preprocess historical data.
    
    Args:
        file_path: Path to historical data file
        
    Returns:
        Tuple of (features, target, preprocessor)
    """
    preprocessor = DataPreprocessor()
    
    # Load and aggregate data
    raw_data = preprocessor.load_historical_data(file_path)
    aggregated_data = preprocessor.aggregate_selection_events(raw_data)
    
    # Create training features
    features, target = preprocessor.create_training_features(aggregated_data)
    
    return features, target, preprocessor


def analyze_historical_patterns(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze historical selection patterns.
    
    Args:
        file_path: Path to historical data file
        
    Returns:
        Analysis results dictionary
    """
    preprocessor = DataPreprocessor()
    
    # Load and process data
    raw_data = preprocessor.load_historical_data(file_path)
    aggregated_data = preprocessor.aggregate_selection_events(raw_data)
    
    # Get insights
    summary = preprocessor.get_data_summary()
    insights = preprocessor.get_category_insights()
    
    return {
        'summary': summary,
        'insights': insights,
        'preprocessor': preprocessor
    }