# Technology Stack

## Core Technologies

### Backend Framework
- **FastAPI**: RESTful API framework
  - Automatic OpenAPI/Swagger documentation
  - Built-in request/response validation with Pydantic
  - Async support for high performance
  - Python type hints integration

### Data Processing
- **Pandas**: Primary data manipulation library
  - DataFrame operations for historical data processing
  - Groupby aggregations for feature engineering
  - Data cleaning and transformation pipelines
  - CSV/Excel file handling

**Real Data Structure**:
- **Historical Training Data**: `present.selection.historic.csv`
  - Columns: employee_shop, employee_branch, employee_gender, product_main_category, product_sub_category, product_brand, product_color, product_durability, product_target_gender, product_utility_type, product_type
- **Classification Schema**: `present.attributes.schema.json`
  - Fields: present_name, present_vendor, model_name, model_no, itemMainCategory, itemSubCategory, color, brand, vendor, valuePrice, targetDemographic, utilityType, durability, usageType
  - Enums: targetDemographic (male/female/unisex), utilityType (practical/work/aesthetic/status/sentimental/exclusive), durability (consumable/durable), usageType (shareable/individual)

### Machine Learning
- **CatBoost**: Gradient boosting framework
  - `CatBoostRegressor` for demand quantity prediction (Poisson loss)
  - Native handling of categorical features
  - Built-in feature importance analysis
  - Cross-validation support
  - Robust to overfitting

- **Scikit-learn**: ML utilities and metrics
  - Model evaluation metrics (MAE, RMSE, R²)
  - Data preprocessing utilities
  - Cross-validation frameworks
  - Model selection tools

### Development Environment
- **Python 3.9+**: Core language version
- **Virtual Environment**: Dependency isolation
- **Git**: Version control with comprehensive `.gitignore`

## Dependencies (Planned)

### Core Dependencies
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pandas>=2.0.0
catboost>=1.2.0  # Updated from xgboost
scikit-learn>=1.3.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
numpy>=1.24.0
apscheduler>=3.10.0
python-multipart>=0.0.6
httpx>=0.24.0
gender-guesser>=0.4.0
optuna>=3.0.0
click>=8.0.0
tabulate>=0.9.0
python-dotenv>=0.21.0  # For explicit .env file loading
```

### Development Dependencies
```
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
jupyter>=1.0.0
```

### Optional Dependencies
```
uvloop>=0.17.0  # Performance (Unix only)
python-multipart>=0.0.6  # File uploads
python-jose[cryptography]>=3.3.0  # JWT tokens
passlib[bcrypt]>=1.7.4  # Password hashing
```

## Technical Constraints

### Performance Requirements
- **API Response Time**: < 2 seconds for prediction requests
- **Throughput**: Support 100+ concurrent requests
- **Memory Usage**: Efficient handling of large datasets
- **Model Size**: Keep trained models under 100MB

### Data Constraints
- **Input Validation**: Strict schema validation for all API inputs
- **Data Types**: Handle mixed categorical and numerical features
- **Missing Data**: Robust handling of incomplete input data
- **Scalability**: Support datasets with 100K+ historical records

### Environment Constraints
- **Python Version**: 3.9+ for modern type hints and features
- **Memory**: Minimum 4GB RAM for model training
- **Storage**: Sufficient space for historical data and models
- **CPU**: Multi-core support for CatBoost training

## Development Setup

### Project Structure
```
predict-presents/
├── src/                 # Source code
├── tests/              # Test files
├── data/               # Data files (gitignored)
├── models/             # Trained models (gitignored)
├── notebooks/          # Jupyter notebooks (gitignored)
├── docs/               # Documentation
├── requirements.txt    # Dependencies
├── .gitignore         # Git ignore rules
└── README.md          # Project documentation
```

### Environment Configuration
```bash
# Virtual environment setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Dependencies installation
pip install -r requirements.txt

# Development tools
pip install -r requirements-dev.txt
```

### Code Quality Tools
- **Black**: Code formatting
- **Flake8**: Linting and style checking
- **MyPy**: Static type checking
- **Pytest**: Unit and integration testing

## Data Flow Technologies

### Input Processing
- **Pydantic Models**: Request/response validation
- **JSON Schema**: API documentation and validation
- **Type Hints**: Runtime type checking

### Data Transformation
- **Pandas DataFrames**: Structured data manipulation
- **NumPy Arrays**: Numerical computations
- **Custom Classifiers**: Gift categorization logic
- **Shop-level Aggregates**: Creation of features like shop diversity and most frequent items.
- **Interaction Features**: Hashing for combined categorical features.

### Model Operations
- **CatBoost API**: Model training and inference
- **CatBoost Model Format / Pickle/Joblib**: Model serialization
- **Scikit-learn Pipelines**: Feature preprocessing

### Output Generation
- **FastAPI Response Models**: Structured API responses
- **JSON Serialization**: Standard data exchange format

## Integration Patterns

### API Integration
- **RESTful Endpoints**: Standard HTTP methods
- **OpenAPI Specification**: Auto-generated documentation
- **Error Handling**: Standardized error responses
- **Request Logging**: Comprehensive request tracking

### Data Integration
- **File-based Input**: CSV, Excel file processing
- **Database Abstraction**: Future database integration
- **Caching Layer**: Model and prediction caching
- **Batch Processing**: Large dataset handling

### ML Integration
- **Model Versioning**: Multiple model support
- **Feature Pipelines**: Consistent feature engineering
- **Prediction Caching**: Performance optimization
- **Model Monitoring**: Performance tracking

## Tool Usage Patterns

### Development Workflow
1. **Code Development**: Local development with FastAPI
2. **Testing**: Pytest for unit and integration tests
3. **Formatting**: Black for consistent code style
4. **Type Checking**: MyPy for type safety
5. **Documentation**: Auto-generated API docs

### Data Science Workflow
1. **Exploration**: Jupyter notebooks for data analysis
2. **Feature Engineering**: Pandas for data transformation
3. **Model Training**: CatBoost for regression models (Poisson)
4. **Evaluation**: Scikit-learn for metrics, Optuna for hyperparameter tuning
5. **Validation**: Cross-validation for model selection

### Deployment Workflow (Future)
1. **Containerization**: Docker for deployment
2. **Orchestration**: Docker Compose for local deployment
3. **Monitoring**: Logging and metrics collection
4. **CI/CD**: Automated testing and deployment

## Security Considerations

### API Security
- **Input Validation**: Strict schema validation
- **Rate Limiting**: Request throttling
- **Error Handling**: No sensitive data in error messages
- **Logging**: Secure logging practices

### Data Security
- **Data Sanitization**: Clean input data
- **Access Control**: Secure data access patterns
- **Configuration**: Environment-based secrets
- **Audit Trail**: Request and response logging

## Performance Optimization

### API Performance
- **Async Operations**: Non-blocking request handling
- **Response Caching**: Cache prediction results
- **Connection Pooling**: Efficient resource usage
- **Compression**: Response compression

### ML Performance
- **Feature Caching**: Cache engineered features
- **Model Loading**: Lazy model loading
- **Batch Predictions**: Efficient batch processing
- **Memory Management**: Optimize memory usage

## Future Technology Considerations

### Scalability
- **Database Integration**: PostgreSQL/MongoDB
- **Message Queues**: Redis/RabbitMQ for async processing
- **Caching**: Redis for distributed caching
- **Load Balancing**: nginx/HAProxy

### Advanced ML
- **MLOps**: MLflow for model management
- **Feature Stores**: Centralized feature management
- **Model Serving**: Dedicated model serving infrastructure
- **A/B Testing**: Model performance comparison

### Monitoring
- **Application Monitoring**: Prometheus/Grafana
- **Log Aggregation**: ELK Stack
- **Error Tracking**: Sentry
- **Performance Monitoring**: APM tools