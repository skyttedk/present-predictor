"""
Data models and enums for the predictive gift selection system.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class TargetDemographic(str, Enum):
    """Target demographic for gifts or gender classification."""
    MALE = "male"
    FEMALE = "female"
    UNISEX = "unisex"


class UtilityType(str, Enum):
    """Utility type of a gift."""
    PRACTICAL = "practical"
    WORK = "work"
    AESTHETIC = "aesthetic"
    STATUS = "status"
    SENTIMENTAL = "sentimental"
    EXCLUSIVE = "exclusive"


class Durability(str, Enum):
    """Durability classification of a gift."""
    CONSUMABLE = "consumable"
    DURABLE = "durable"


class UsageType(str, Enum):
    """Usage type of a gift."""
    SHAREABLE = "shareable"
    INDIVIDUAL = "individual"


class ClassifiedGift(BaseModel):
    """Gift with classified attributes."""
    item_main_category: str = Field(..., description="Main category of the item")
    item_sub_category: str = Field(..., description="Sub category of the item")
    color: str = Field(default="NONE", description="Color of the item")
    brand: str = Field(..., description="Brand of the item")
    vendor: Optional[str] = Field(default=None, description="Vendor of the item")
    value_price: Optional[float] = Field(default=None, description="Price value of the item")
    target_demographic: TargetDemographic = Field(..., description="Target demographic")
    utility_type: UtilityType = Field(..., description="Utility type")
    durability: Durability = Field(..., description="Durability classification")
    usage_type: UsageType = Field(..., description="Usage type")


class HistoricalGiftRecord(BaseModel):
    """Historical gift selection record format."""
    employee_shop: str
    employee_branch: str
    employee_gender: str
    product_main_category: str
    product_sub_category: str
    product_brand: str
    product_color: str = "NONE"
    product_durability: str
    product_target_gender: str
    product_utility_type: str
    product_type: str