Project Overview: Predictive Gift Selection System for Gavefabrikken
Business Context
Gavefabrikken operates a B2B gift distribution service where companies provide curated gift selections to their employees through dedicated online portals (Gift Shops). Each company receives access to a customized portal featuring a subset of gifts selected from Gavefabrikken's extensive catalog. During seasonal periods, particularly Christmas, employees access their company's portal to select their preferred gifts.
Current Challenge
The primary operational challenge lies in demand forecasting. Gavefabrikken must predict gift selection quantities both pre-season and during active periods to ensure optimal inventory management. Currently, this process relies on manual estimation combined with basic statistical averages, resulting in:

Inventory imbalances (overstocking and stockouts)
Significant post-season operational overhead
Increased costs from back-orders and surplus gift returns
Suboptimal customer satisfaction due to unavailable selections

Proposed Solution
Development of a machine learning-powered demand prediction system utilizing historical gift selection data to generate accurate quantity forecasts.
Technical Architecture
Data Pipeline:

Historical Data Analysis: Process multi-year gift selection records containing:

Employee demographics (shop_id, branch_code, gender)
Gift characteristics (Item Main Category, Item Sub Category, Color, Brand, Target Demographics, Utility Type, Durability, Usage Type)


API Integration: RESTful service accepting company-specific requests
Data Classification: Automated extraction and categorization
Prediction Engine: Machine learning model returning product-specific demand forecasts

API Request Flow:
Step 1 - Initial API Request:
{
"branch_no" : "",
"gifts" : [
{
"product_id" : "",
"description" : "descr+model+variant"
}
],
"employees" : [
{
"Name" : ""
}
]
}
Step 2 - Internal Data Reclassification:
{
"branch_no" : "",
"gifts" : [
{
"Item Main Category" : "",
"Item Sub Category" : "",
"Color" : "",
"Brand" : "",
"Target Demographics" : "",
"Utility Type" : "",
"Durability" : "",
"Usage Type" : ""
}
],
"employees" : [
{
"gender" : ""
}
]
}
Step 3 - API Response:
[
{
"product_id" : "",
"expected_qty" : 0
}
]
Implementation Approach
Technology Stack:
Python-based solution leveraging:

Data Processing: Pandas for aggregation and grouping operations
Machine Learning: XGBoost Regressor for demand prediction
Model Evaluation: Scikit-learn metrics for performance assessment

Key Development Notes:
Data Aggregation Example:
final_df = structured_data.groupby(['date', 'category', 'product_base', 'color', 'type', 'size']).agg({'qty_sold': 'sum'}).reset_index()
Model Training Hints:

Use XGBRegressor
Focus on metrics evaluation

Development Phases:

Historical data preprocessing and feature engineering
Exploratory data analysis with focus on aggregation patterns
Model training and validation using XGBoost regression
API development for real-time prediction delivery
Performance monitoring and model refinement

Expected Outcomes

Reduced inventory management costs through improved demand accuracy
Minimized post-season operational complexity
Enhanced customer satisfaction via better gift availability
Data-driven decision making for seasonal planning

This solution transforms Gavefabrikken's reactive inventory approach into a proactive, analytics-driven system capable of scaling with business growth while maintaining operational efficiency.