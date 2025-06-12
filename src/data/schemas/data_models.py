"""
Core data models for the Predictive Gift Selection System.
Defines data structures for historical data, classification schema, and processing pipeline.
"""

from typing import List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator
import pandas as pd


class TargetDemographic(str, Enum):
    """Gender demographic enumeration."""
    MALE = "male"
    FEMALE = "female"
    UNISEX = "unisex"


class UtilityType(str, Enum):
    """Utility type enumeration."""
    PRACTICAL = "practical"
    WORK = "work"
    AESTHETIC = "aesthetic"
    STATUS = "status"
    SENTIMENTAL = "sentimental"
    EXCLUSIVE = "exclusive"


class Durability(str, Enum):
    """Durability enumeration."""
    CONSUMABLE = "consumable"
    DURABLE = "durable"


class UsageType(str, Enum):
    """Usage type enumeration."""
    SHAREABLE = "shareable"
    INDIVIDUAL = "individual"


class HistoricalRecord(BaseModel):
    """
    Model for historical gift selection data records.
    Matches the structure of present.selection.historic.csv
    """
    
    employee_shop: str = Field(..., description="Shop identifier")
    employee_branch: str = Field(..., description="Branch identifier")
    employee_gender: TargetDemographic = Field(..., description="Employee gender")
    
    product_main_category: str = Field(..., description="Primary product category")
    product_sub_category: str = Field(..., description="Product subcategory")
    product_brand: str = Field(..., description="Product brand")
    product_color: str = Field(..., description="Product color (NONE if not applicable)")
    
    product_durability: Durability = Field(..., description="Product durability type")
    product_target_gender: TargetDemographic = Field(..., description="Target gender demographic")
    product_utility_type: UtilityType = Field(..., description="Product utility purpose")
    product_type: UsageType = Field(..., description="Product usage type")
    
    class Config:
        use_enum_values = True
        
    def to_dict(self) -> dict:
        """Convert to dictionary for pandas DataFrame creation."""
        return {
            "employee_shop": self.employee_shop,
            "employee_branch": self.employee_branch,
            "employee_gender": self.employee_gender.value,
            "product_main_category": self.product_main_category,
            "product_sub_category": self.product_sub_category,
            "product_brand": self.product_brand,
            "product_color": self.product_color,
            "product_durability": self.product_durability.value,
            "product_target_gender": self.product_target_gender.value,
            "product_utility_type": self.product_utility_type.value,
            "product_type": self.product_type.value,
        }


class GiftAttributes(BaseModel):
    """
    Model for API classification schema.
    Matches the structure of product.attributes.schema.json
    """
    
    itemMainCategory: str = Field(..., description="Primary category of the item")
    itemSubCategory: str = Field(..., description="Subcategory providing more specific classification")
    color: str = Field(..., description="Primary color of the item, NONE if not applicable")
    brand: str = Field(..., description="Brand name of the item, NONE if not applicable")
    vendor: str = Field(..., description="Vendor or supplier, NONE if not applicable")
    valuePrice: float = Field(..., description="Monetary value or price point")
    
    targetDemographic: TargetDemographic = Field(..., description="Primary gender demographic")
    utilityType: UtilityType = Field(..., description="Primary utility or purpose")
    durability: Durability = Field(..., description="Product durability type")
    usageType: UsageType = Field(..., description="Usage pattern")
    
    class Config:
        use_enum_values = True
        
    def to_historical_format(self, employee_shop: str, employee_branch: str, employee_gender: TargetDemographic) -> HistoricalRecord:
        """
        Convert API classification to historical data format.
        Maps fields from API schema to CSV column structure.
        """
        return HistoricalRecord(
            employee_shop=employee_shop,
            employee_branch=employee_branch,
            employee_gender=employee_gender,
            product_main_category=self.itemMainCategory,
            product_sub_category=self.itemSubCategory,
            product_brand=self.brand,
            product_color=self.color,
            product_durability=self.durability,
            product_target_gender=self.targetDemographic,
            product_utility_type=self.utilityType,
            product_type=self.usageType,
        )


class EmployeeData(BaseModel):
    """Model for employee information."""
    
    name: str = Field(..., description="Employee full name")
    gender: Optional[TargetDemographic] = Field(None, description="Classified gender (derived from name)")
    
    class Config:
        use_enum_values = True


class ProcessedEmployee(BaseModel):
    """Model for processed employee data with classified gender."""
    
    name: str = Field(..., description="Employee full name")
    gender: TargetDemographic = Field(..., description="Classified gender")
    
    class Config:
        use_enum_values = True


class ClassifiedGift(BaseModel):
    """
    Model for gifts that have been processed through classification.
    Combines original product info with classified attributes.
    """
    
    product_id: str = Field(..., description="Original product identifier")
    description: str = Field(..., description="Original product description")
    attributes: GiftAttributes = Field(..., description="Classified gift attributes")
    confidence_score: Optional[float] = Field(None, description="Classification confidence (0-1)")
    
    def to_historical_format(self, employee_shop: str, employee_branch: str, employee_gender: TargetDemographic) -> HistoricalRecord:
        """Convert to historical data format for ML processing."""
        return self.attributes.to_historical_format(employee_shop, employee_branch, employee_gender)


class DatasetSample(BaseModel):
    """
    Model for a complete data sample including employee and gift information.
    Used for ML training and prediction.
    """
    
    employee_shop: str = Field(..., description="Shop identifier")
    employee_branch: str = Field(..., description="Branch identifier")
    employee: ProcessedEmployee = Field(..., description="Processed employee data")
    gift: ClassifiedGift = Field(..., description="Classified gift data")
    quantity: Optional[int] = Field(None, description="Selected quantity (for training data)")
    
    def to_historical_record(self) -> HistoricalRecord:
        """Convert to historical record format."""
        return self.gift.to_historical_format(
            employee_shop=self.employee_shop,
            employee_branch=self.employee_branch,
            employee_gender=self.employee.gender
        )


class PredictionFeatures(BaseModel):
    """
    Model for feature vectors used in ML prediction.
    Contains all features needed for XGBoost model.
    """
    
    # Employee features
    employee_shop: str = Field(..., description="Shop identifier")
    employee_branch: str = Field(..., description="Branch identifier")
    employee_gender: TargetDemographic = Field(..., description="Employee gender")
    
    # Product features
    product_main_category: str = Field(..., description="Primary product category")
    product_sub_category: str = Field(..., description="Product subcategory")
    product_brand: str = Field(..., description="Product brand")
    product_color: str = Field(..., description="Product color")
    product_durability: Durability = Field(..., description="Product durability")
    product_target_gender: TargetDemographic = Field(..., description="Target gender")
    product_utility_type: UtilityType = Field(..., description="Utility type")
    product_type: UsageType = Field(..., description="Usage type")
    
    # Additional computed features (can be added later)
    category_popularity: Optional[float] = Field(None, description="Category popularity score")
    brand_preference: Optional[float] = Field(None, description="Brand preference score")
    gender_match_score: Optional[float] = Field(None, description="Gender matching score")
    
    class Config:
        use_enum_values = True
        
    @classmethod
    def from_historical_record(cls, record: HistoricalRecord) -> "PredictionFeatures":
        """Create prediction features from historical record."""
        return cls(
            employee_shop=record.employee_shop,
            employee_branch=record.employee_branch,
            employee_gender=record.employee_gender,
            product_main_category=record.product_main_category,
            product_sub_category=record.product_sub_category,
            product_brand=record.product_brand,
            product_color=record.product_color,
            product_durability=record.product_durability,
            product_target_gender=record.product_target_gender,
            product_utility_type=record.product_utility_type,
            product_type=record.product_type,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for ML processing."""
        return {
            "employee_shop": self.employee_shop,
            "employee_branch": self.employee_branch,
            "employee_gender": self.employee_gender.value,
            "product_main_category": self.product_main_category,
            "product_sub_category": self.product_sub_category,
            "product_brand": self.product_brand,
            "product_color": self.product_color,
            "product_durability": self.product_durability.value,
            "product_target_gender": self.product_target_gender.value,
            "product_utility_type": self.product_utility_type.value,
            "product_type": self.product_type.value,
            "category_popularity": self.category_popularity,
            "brand_preference": self.brand_preference,
            "gender_match_score": self.gender_match_score,
        }


def load_historical_data(file_path: str) -> List[HistoricalRecord]:
    """
    Load and validate historical data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of validated HistoricalRecord objects
        
    Raises:
        ValueError: If data validation fails
    """
    try:
        df = pd.read_csv(file_path)
        records = []
        
        for _, row in df.iterrows():
            record = HistoricalRecord(
                employee_shop=str(row['employee_shop']),
                employee_branch=str(row['employee_branch']),
                employee_gender=row['employee_gender'],
                product_main_category=row['product_main_category'],
                product_sub_category=row['product_sub_category'],
                product_brand=row['product_brand'],
                product_color=row['product_color'],
                product_durability=row['product_durability'],
                product_target_gender=row['product_target_gender'],
                product_utility_type=row['product_utility_type'],
                product_type=row['product_type']
            )
            records.append(record)
            
        return records
        
    except Exception as e:
        raise ValueError(f"Failed to load historical data: {e}")


def create_dataframe_from_records(records: List[HistoricalRecord]) -> pd.DataFrame:
    """
    Convert list of HistoricalRecord objects to pandas DataFrame.
    
    Args:
        records: List of HistoricalRecord objects
        
    Returns:
        Pandas DataFrame with historical data
    """
    data = [record.to_dict() for record in records]
    return pd.DataFrame(data)