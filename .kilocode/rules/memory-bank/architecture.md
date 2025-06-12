# System Architecture

## Overview
The Predictive Gift Selection System follows a modular architecture with clear separation of concerns between data processing, machine learning, and API layers.

## High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Gateway   │────│  Data Pipeline   │────│ Prediction ML   │
│   (FastAPI)     │    │   (Pandas)       │    │  (XGBoost)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌──────────────────┐              │
         └──────────────│ Data Storage     │──────────────┘
                        │ (Files/Database) │
                        └──────────────────┘
```

## Core Components

### 1. API Layer
**Location**: `/src/api/`
- **Framework**: FastAPI for RESTful API
- **Responsibilities**:
  - Request validation and parsing
  - API endpoint management
  - Response formatting
  - Error handling

**Key Files**:
- `main.py` - FastAPI application entry point
- `endpoints/` - API route definitions
- `schemas/` - Pydantic models for request/response validation
- `middleware/` - Request/response processing

### 2. Data Pipeline Layer
**Location**: `/src/data/`
- **Framework**: Pandas for data manipulation
- **Responsibilities**:
  - Historical data preprocessing
  - Gift classification and feature extraction
  - Employee demographic processing
  - Data aggregation and transformation

**Key Files**:
- `preprocessor.py` - Data cleaning and transformation
- `classifier.py` - Gift categorization logic
- `aggregator.py` - Data grouping and aggregation
- `schemas/` - Data structure definitions

### 3. Machine Learning Layer
**Location**: `/src/ml/`
- **Framework**: XGBoost for regression
- **Responsibilities**:
  - Model training and validation
  - Feature engineering
  - Prediction generation
  - Model performance monitoring

**Key Files**:
- `model.py` - XGBoost model implementation
- `features.py` - Feature engineering pipeline
- `trainer.py` - Model training orchestration
- `predictor.py` - Prediction service
- `evaluation.py` - Model metrics and validation

### 4. Configuration Layer
**Location**: `/src/config/`
- **Responsibilities**:
  - Environment configuration
  - Model hyperparameters
  - API settings
  - Database connections

## Data Flow Architecture

### Real Data Structure Analysis

**Historical Training Data** (`present.selection.historic.csv`):
```
employee_shop, employee_branch, employee_gender,
product_main_category, product_sub_category, product_brand,
product_color, product_durability, product_target_gender,
product_utility_type, product_type
```

**Classification Schema** (`product.attributes.schema.json`):
```json
{
  "itemMainCategory": "string",
  "itemSubCategory": "string",
  "color": "string",
  "brand": "string",
  "vendor": "string",
  "valuePrice": "number",
  "targetDemographic": "male|female|unisex",
  "utilityType": "practical|work|aesthetic|status|sentimental|exclusive",
  "durability": "consumable|durable",
  "usageType": "shareable|individual"
}
```

### Three-Step API Processing Pipeline

```
Step 1: Raw Request Processing
├── Input: {branch_no, gifts[{product_id, description}], employees[{name}]}
├── Validation: Request schema validation
└── Output: Validated raw data

Step 2: Data Reclassification
├── Input: Raw validated data
├── Processing:
│   ├── Gift description → JSON schema attributes (LLM/rule-based)
│   ├── Employee name → gender classification
│   └── Field mapping: JSON schema → CSV column names
└── Output: Classified feature data matching historical structure

Step 3: Prediction Generation
├── Input: Classified feature data (CSV format)
├── Processing: XGBoost model inference using historical patterns
└── Output: {product_id, expected_qty}[]
```

### Data Field Mapping
```
API Classification → Historical Training Data
itemMainCategory → product_main_category
itemSubCategory → product_sub_category
color → product_color
brand → product_brand
targetDemographic → product_target_gender
utilityType → product_utility_type
durability → product_durability
usageType → product_type
vendor → (not in historical data)
valuePrice → (not in historical data)
```

## Key Technical Decisions

### Data Processing Strategy
- **Pandas-based pipeline** for structured data manipulation
- **Groupby aggregation** pattern for historical data analysis
- **Feature engineering** focused on categorical variables
- **Real-time classification** for incoming requests

### Machine Learning Approach
- **XGBoost Regressor** for demand prediction
- **Scikit-learn metrics** for model evaluation
- **Historical aggregation** as primary feature source
- **Cross-validation** for model selection

### API Design Patterns
- **RESTful architecture** with clear resource endpoints
- **Request/Response validation** using Pydantic schemas
- **Error handling** with standardized error responses
- **Stateless design** for scalability

## Component Relationships

### Data Dependencies
```
Historical Data → Preprocessor → Feature Engineering → Model Training
                              ↘                    ↗
API Request → Classifier → Feature Extraction → Prediction
```

### Service Integration
- **API Layer** calls **Data Pipeline** for request processing
- **Data Pipeline** feeds **ML Layer** for feature preparation
- **ML Layer** returns predictions to **API Layer**
- **Configuration Layer** provides settings to all components

## Source Code Structure (Planned)

```
src/
├── api/
│   ├── main.py              # FastAPI app entry point
│   ├── endpoints/
│   │   └── predictions.py   # Prediction endpoints
│   ├── schemas/
│   │   ├── requests.py      # API request models
│   │   └── responses.py     # API response models
│   └── middleware/
│       └── validation.py    # Request validation
├── data/
│   ├── preprocessor.py      # Data cleaning pipeline
│   ├── classifier.py        # Gift categorization
│   ├── aggregator.py        # Data aggregation logic
│   └── schemas/
│       └── data_models.py   # Data structure definitions
├── ml/
│   ├── model.py             # XGBoost model wrapper
│   ├── features.py          # Feature engineering
│   ├── trainer.py           # Model training pipeline
│   ├── predictor.py         # Prediction service
│   └── evaluation.py        # Model metrics
├── config/
│   ├── settings.py          # Application configuration
│   └── model_config.py      # ML model parameters
└── utils/
    ├── logging.py           # Logging configuration
    └── exceptions.py        # Custom exception classes
```

## Critical Implementation Paths

### 1. Data Aggregation Pattern
```python
# Core aggregation logic from requirements
final_df = structured_data.groupby([
    'date', 'category', 'product_base', 
    'color', 'type', 'size'
]).agg({'qty_sold': 'sum'}).reset_index()
```

### 2. Model Training Flow
```python
# XGBoost training pattern
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train)
# Focus on metrics evaluation
```

### 3. API Processing Chain
```python
# Three-step processing pattern
raw_data = validate_request(request)
classified_data = classify_and_transform(raw_data)
predictions = generate_predictions(classified_data)
```

## Scalability Considerations

- **Stateless API design** for horizontal scaling
- **Modular component structure** for independent scaling
- **Caching strategy** for model predictions
- **Async processing** for batch operations
- **Database abstraction** for storage flexibility

## Deployment Architecture (Future)

- **Containerized services** using Docker
- **API Gateway** for request routing
- **Load balancing** for high availability
- **Model versioning** for A/B testing
- **Monitoring and logging** for operational visibility