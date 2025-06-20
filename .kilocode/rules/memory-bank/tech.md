# Technology Stack

## Core Technologies

### Backend Framework
- **FastAPI**: RESTful API framework
  - Automatic OpenAPI/Swagger documentation via Pydantic integration.
  - Built-in request/response validation with Pydantic.
  - Async support for high performance.
  - Python type hints integration.

### Data Processing
- **Pandas**: Primary data manipulation library
  - DataFrame operations for historical data processing.
  - Groupby aggregations for feature engineering.
  - Data cleaning and transformation pipelines.
  - CSV file handling for historical data.

### Machine Learning
- **CatBoost**: Gradient boosting framework
  - `CatBoostRegressor` for demand quantity prediction (using Poisson loss as per [`src/ml/catboost_trainer.py`](src/ml/catboost_trainer.py:1) and [`src/config/settings.py`](src/config/settings.py:105)).
  - Native handling of categorical features.
  - Built-in feature importance analysis.
  - Cross-validation support.
- **Scikit-learn**: ML utilities and metrics
  - Model evaluation metrics (R², MAE, RMSE).
  - Data preprocessing utilities (e.g., `FeatureHasher`).
  - Cross-validation frameworks (`StratifiedKFold`).
- **Optuna**: Hyperparameter optimization framework (used with CatBoost in [`src/ml/catboost_trainer.py`](src/ml/catboost_trainer.py:1)).

### Configuration Management
- **Pydantic & Pydantic-Settings**: For application and model configuration, loading from environment variables and `.env` files (see [`src/config/settings.py`](src/config/settings.py:1)).

### Development Environment
- **Python**: Version >=3.9 (as per [`pyproject.toml`](pyproject.toml:1) and [`src/config/settings.py`](src/config/settings.py:1) type hinting).
- **Virtual Environment**: Standard practice (e.g., venv, conda).
- **Git**: Version control.

## Dependencies

### Core Dependencies (from [`requirements.txt`](requirements.txt:1))
- `fastapi>=0.104.0`
- `uvicorn[standard]>=0.24.0`
- `pandas>=2.0.0`
- `scikit-learn>=1.3.0`
- `pydantic>=2.0.0`
- `pydantic-settings>=2.0.0`
- `numpy>=1.24.0`
- `apscheduler>=3.10.0` (Note: Usage not immediately evident from current task scope)
- `python-multipart>=0.0.6`
- `httpx>=0.24.0`
- `gender-guesser>=0.4.0`
- `catboost>=1.2.0`
- `optuna>=3.0.0`
- `click>=8.0.0` (Note: Usage not immediately evident, possibly for CLI scripts)
- `tabulate>=0.9.0` (Note: Usage not immediately evident, possibly for CLI scripts/reports)
- `python-dotenv>=0.21.0`
- `gunicorn>=20.1.0` (Production server)
- `psycopg2-binary>=2.9.5` (PostgreSQL support for Heroku)

**Note on Dependency Discrepancy:**
- [`requirements.txt`](requirements.txt:1) lists `catboost>=1.2.0`.
- [`pyproject.toml`](pyproject.toml:1) lists `xgboost>=1.7.0` under `[project.dependencies]` but does *not* list `catboost`. This should be reconciled for consistency. The actual model in use is CatBoost.

### Development Dependencies (from [`pyproject.toml`](pyproject.toml:1))
- `pytest>=7.0.0`
- `pytest-asyncio>=0.21.0`
- `pytest-cov>=4.0.0`
- `black>=23.0.0`
- `flake8>=6.0.0`
- `mypy>=1.0.0`
- `pre-commit>=3.0.0`
- `jupyter>=1.0.0`
- `httpx>=0.24.0` (also in core)

## Technical Constraints (Inferred & from Settings)

- **Performance Requirements**: API response time, throughput (general goals).
- **Data Constraints**: Input validation via Pydantic, handling mixed data types, missing data strategies.
- **Environment Constraints**: Python 3.9+.
- **Model Artifacts**: CatBoost models stored in `.cbm` format, metadata and parameters often pickled (as seen in [`src/ml/catboost_trainer.py`](src/ml/catboost_trainer.py:511)).

## Development Setup & Tooling (from [`pyproject.toml`](pyproject.toml:1) and `README.md`)

### Project Structure
```
src/
├── api/                 # FastAPI application and endpoints
├── data/               # Data processing and classification
├── ml/                 # Machine learning models and features
├── config/             # Application and model configuration
└── utils/              # Logging, exceptions, and utilities
tests/
# ... and other standard project directories like models/, docs/, etc.
```

### Environment Configuration
- Standard Python virtual environment setup.
- Dependencies installed via `pip install -r requirements.txt` and `pip install -r requirements-dev.txt`.

### Code Quality Tools
- **Black**: Code formatting.
- **Flake8**: Linting and style checking.
- **MyPy**: Static type checking.
- **Pytest**: Unit and integration testing, with coverage.
- **Pre-commit**: Git hooks for automated checks.

## Data Flow Technologies

### Input Processing
- **Pydantic Models**: For API request/response validation.
- **JSON**: Standard data exchange format for API.
- **OpenAI API**: Used for gift classification (as per [`src/config/settings.py`](src/config/settings.py:94) and `OpenAISettings`). Assistant ID: `asst_BuFvA6iXF4xSyQ4px7Q5zjiN`.

### Data Transformation
- **Pandas DataFrames**: For structured data manipulation.
- **NumPy Arrays**: For numerical computations.
- **`gender_guesser`**: For employee gender classification.
- **`ShopFeatureResolver`** ([`src/ml/shop_features.py`](src/ml/shop_features.py:1)): For shop-specific feature engineering.
- **`FeatureHasher`** (Scikit-learn): For creating interaction features.

### Model Operations
- **CatBoost API**: For model training (`CatBoostRegressor.fit()`) and inference (`CatBoostRegressor.predict()`, `CatBoostRegressor.load_model()`).
- **CatBoost `Pool`**: Optimized data structure for CatBoost.
- **Pickle/Joblib**: Likely used for saving/loading model metadata, parameters, and other artifacts (standard Python practice, confirmed in trainer script).

### Output Generation
- **FastAPI Response Models**: For structured API responses.
- **JSON Serialization**: For API output.

## Integration Patterns

### API Integration
- **RESTful Endpoints**: Standard HTTP methods via FastAPI.
- **OpenAPI Specification**: Auto-generated by FastAPI.

### ML Integration
- **Singleton Pattern**: For predictor instance to efficiently load and reuse the model ([`src/ml/predictor.py`](src/ml/predictor.py:403)).
- **Feature Pipelines**: Ensuring consistency between training ([`src/ml/catboost_trainer.py`](src/ml/catboost_trainer.py:1)) and inference ([`src/ml/predictor.py`](src/ml/predictor.py:1)) feature engineering steps.
- **Model Versioning**: Implied by model storage paths and metadata, though formal versioning system not detailed.

## Security Considerations (from [`src/config/settings.py`](src/config/settings.py:81))
- **API Security**: Secret key, JWT for access tokens, rate limiting.
- **Configuration**: Environment-based secrets (e.g., `OPENAI_API_KEY`).