# System Architecture

## Overview
The Predictive Gift Selection System follows a modular architecture with clear separation of concerns between data processing, machine learning, and API layers.

## High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│   API Gateway   │────│  Data Pipeline   │────│ Two-Stage Prediction │
│   (FastAPI)     │    │   (Pandas)       │    │    (CatBoost)        │
└─────────────────┘    └──────────────────┘    └──────────────────────┘
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
- `main.py` - FastAPI application entry point (includes `/test`, `/addPresent`, `/addPresentsProcessed`, `/deleteAllPresents`, scheduler initialization)
- `endpoints/` - API route definitions (e.g., `/predict`)
- `schemas/` - Pydantic models for request/response validation (e.g., `AddPresentRequest`, `CSVImportResponse`, `DeleteAllPresentsResponse`)
- `middleware/` - Request/response processing
- `scheduler/tasks.py` - Background task for automatic present classification

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
- `openai_client.py` - OpenAI Assistant API integration for present classification
- `gender_classifier.py` - Enhanced gender classification with Danish name support
- `aggregator.py` - Data grouping and aggregation
- `schemas/` - Data structure definitions

### 6. Database Layer (New)
**Location**: `/src/database/`
- **Framework**: PostgreSQL with psycopg2
- **Responsibilities**:
  - Database connections and query execution
  - CSV import with bulk operations (500x performance improvement)
  - User authentication and API key management
  - Present attribute caching and management

**Key Files**:
- `csv_import.py` - Optimized bulk CSV import functionality (3 seconds vs 26 minutes)
- `db_postgres.py` - PostgreSQL connection management
- `db_factory.py` - Database abstraction layer
- `users.py` - User authentication system
- `presents.py` - Present attribute management
- `api_logs.py` - API request/response logging

### 3. Machine Learning Layer
**Location**: `/src/ml/`
- **Framework**: CatBoost with Two-Stage Architecture
- **Responsibilities**:
  - Two-stage model: Binary classification + Count regression
  - Feature engineering (11 base + shop features + new share/rank features)
  - Native categorical handling
  - Model performance monitoring

**Key Files**:
- `model.py` - CatBoost model implementations (classifier & regressor)
- `two_stage.py` - Two-stage prediction pipeline
- `features.py` - Feature engineering (base, shop-level, share/rank features)
- `trainer.py` - Model training orchestration
- `predictor.py` - Combined prediction service
- `evaluation.py` - Model metrics and validation

### 4. Configuration Layer
**Location**: `/src/config/`
- **Responsibilities**:
  - Environment configuration
  - Model hyperparameters
  - API settings
  - Database connections
  - Scheduler configuration

### 5. Scheduler Layer
**Location**: Integrated into FastAPI app
- **Framework**: APScheduler
- **Responsibilities**:
  - Periodic execution of classification tasks
  - Background processing of pending presents
  - Error handling and retry logic

## Data Flow Architecture

### Real Data Structure Analysis

**Historical Training Data** (`present.selection.historic.csv`):
```
employee_shop, employee_branch, employee_gender,
product_main_category, product_sub_category, product_brand,
product_color, product_durability, product_target_gender,
product_utility_type, product_type
```

**Classification Schema** (`present.attributes.schema.json`):
```json
{
  "present_name": "string",
  "present_vendor": "string",
  "model_name": "string",
  "model_no": "string",
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

### Five-Step API Processing Pipeline (Two-Stage Model)

```
Step 0a: Add Present (Optional, via /addPresent endpoint)
├── Input: {present_name, present_vendor, model_name, model_no}
├── Processing: Calculate MD5 hash, check if exists in `present_attributes`.
│             If not, add to `present_attributes` with 'pending_classification' status.
└── Output: Confirmation or existing present details.

Step 0b: Bulk CSV Import (Optional, via /addPresentsProcessed endpoint)
├── Input: CSV file with pre-classified present attributes
├── Processing:
│   ├── Parse CSV with robust error handling
│   ├── Bulk existence check using IN clause (1 query vs thousands)
│   ├── Batch insert using executemany() (1 operation vs thousands)
│   └── Single transaction for entire operation
└── Output: Import statistics (imported, skipped, errors, processing time)

Step 0c: Delete All Presents (Testing, via /deleteAllPresents endpoint)
├── Input: Authenticated request
├── Processing: DELETE FROM present_attributes
└── Output: Deletion count and processing time

Step 1: Raw Request Processing (for /predict endpoint)
├── Input: {branch_no, gifts[{present_name, present_vendor, model_name, model_no}], employees[{name}]}
├── Validation: Request schema validation
└── Output: Validated raw data

Step 2: Data Reclassification
├── Input: Validated raw data (gifts will be looked up in `present_attributes` or classified if status is 'pending_classification' or missing)
├── Processing:
│   ├── Gift details (name, vendor, model, model_no) → JSON schema attributes (OpenAI Assistant API)
│   ├── Employee name → gender classification (Enhanced gender_guesser with Danish names)
│   └── Field mapping: JSON schema → CSV column names
└── Output: Classified feature data matching historical structure

Step 2.5: Background Classification (Scheduler)
├── Every 2 minutes: APScheduler triggers `fetch_pending_present_attributes`
├── Queries database for presents with 'pending_classification' status
├── For each pending present:
│   ├── Send to OpenAI Assistant API
│   ├── Parse classification response
│   └── Update database with classified attributes
└── Error handling: Mark as 'error_openai_api' if classification fails

Step 3: Two-Stage Prediction
├── Input: Classified feature data (CSV format)
├── Stage 1: Binary classification - Will select? (CatBoostClassifier)
├── Stage 2: Count regression - How many? (CatBoostRegressor with Poisson)
└── Combined: P(select) × Expected_count

Step 4: Response Generation
├── Input: Combined predictions with confidence intervals
├── Processing: Format results with uncertainty estimates
└── Output: {product_id, expected_qty, confidence_interval}[]
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
- **CatBoost Two-Stage Model** for count prediction
  - Stage 1: Binary classification (selection probability)
  - Stage 2: Poisson regression (count if selected)
- **Native categorical handling** for high-cardinality features
- **No log transformation** needed with Poisson loss
- **Enhanced features**: Shop shares, ranks, interactions
- **Stratified + Leave-One-Shop-Out CV** for robust evaluation

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
API Request (/predict) → Classifier → Feature Extraction → Prediction
API Request (/addPresent) → Add to `present_attributes` (for future classification/lookup)
API Request (/addPresentsProcessed) → Bulk CSV Import (OPTIMIZED: 500x faster)
API Request (/deleteAllPresents) → Clear all data (for testing)
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
│   │   └── predictions.py   # Prediction endpoints (e.g. /predict)
│   │   └── presents.py      # Present management endpoints (e.g. /addPresent - though currently in main.py)
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
│   ├── model.py             # CatBoost model wrappers
│   ├── two_stage.py         # Two-stage prediction logic
│   ├── features.py          # Feature engineering (enhanced)
│   ├── trainer.py           # Model training pipeline
│   ├── predictor.py         # Combined prediction service
│   └── evaluation.py        # Model metrics (MAPE, RMSE, R²)
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

### 2. Two-Stage Model Training Flow
```python
# Stage 1: Binary Classification
from catboost import CatBoostClassifier
classifier = CatBoostClassifier(
    iterations=500,
    cat_features=categorical_cols,
    random_state=42
)
classifier.fit(X_train, y_binary, eval_set=(X_val, y_val_binary))

# Stage 2: Count Regression (positives only)
from catboost import CatBoostRegressor
regressor = CatBoostRegressor(
    iterations=1000,
    loss_function='Poisson',  # No log transform needed
    cat_features=categorical_cols,
    random_state=42
)
# Train only on positive samples
X_train_pos = X_train[y_train > 0]
y_train_pos = y_train[y_train > 0]
regressor.fit(X_train_pos, y_train_pos)
```

### 3. API Processing Chain with Two-Stage Prediction
```python
# Four-step processing pattern
raw_data = validate_request(request)
classified_data = classify_and_transform(raw_data)

# Two-stage prediction
selection_probs = classifier.predict_proba(classified_data)[:, 1]
expected_counts = regressor.predict(classified_data)
final_predictions = selection_probs * expected_counts

# Add confidence intervals
predictions = add_confidence_intervals(final_predictions)
```

### 4. Enhanced Feature Engineering
```python
# New shop-level share features
df['product_share_in_shop'] = (
    df.groupby(['employee_shop', 'product_id'])['selection_count'].transform('sum') /
    df.groupby('employee_shop')['selection_count'].transform('sum')
)

# Rank features
df['product_rank_in_shop'] = df.groupby('employee_shop')['selection_count'].rank(
    method='dense', ascending=False
)

# Interaction hashing (memory efficient)
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=32, input_type='string')
interactions = hasher.transform(
    df[['employee_branch', 'product_main_category']].apply(
        lambda x: f"{x[0]}_{x[1]}", axis=1
    )
)
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