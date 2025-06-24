# Product Vision: Predictive Gift Selection System

## Why This Project Exists

Gavefabrikken operates a B2B gift distribution service where companies provide curated gift selections to their employees through dedicated online portals (Gift Shops). The primary operational challenge lies in accurately forecasting demand for these gifts, especially during peak seasonal periods like Christmas. Inaccurate forecasting leads to significant business inefficiencies.

## Problems We Solve

The project aims to address the following critical pain points stemming from manual and basic statistical demand forecasting:

-   **Inventory Imbalances:** Overstocking of unpopular gifts and stockouts of desired items, leading to wasted resources and missed opportunities.
-   **Operational Overhead:** Significant manual effort and complexity in managing inventory post-season, including handling returns and back-orders.
-   **Increased Costs:** Financial impact from expedited shipping for back-orders, storage of surplus gifts, and potential loss from unsellable returned items.
-   **Suboptimal Customer Satisfaction:** Employees of client companies may be unable to select their preferred gifts due to unavailability, negatively impacting their experience and, by extension, the client's satisfaction with Gavefabrikken's service.

## How It Should Work

The system is designed as a machine learning-powered demand prediction engine that integrates into Gavefabrikken's operational workflow.

### Core Workflow:

1.  **Data Input:** The system ingests historical gift selection data, which includes employee demographics (shop, branch, gender) and detailed gift characteristics (categories, brand, color, utility, durability, etc.). For real-time predictions, it accepts company-specific requests detailing the current gift assortment, employee roster, and company context (e.g., CVR, branch).
2.  **Data Processing & Classification:**
    *   Historical data is preprocessed and aggregated to identify patterns.
    *   For new prediction requests, input data (gift descriptions, employee names) is classified into structured features (e.g., gift attributes via OpenAI, employee gender via `gender_guesser`).
3.  **Prediction Generation (Selection Rate):** A machine learning model (CatBoost Regressor) predicts a `selection_rate` (a value between 0 and 1) for each gift, considering employee gender groups.
4.  **Prediction Scaling & Aggregation:** The predicted `selection_rate` for each gift-gender combination is scaled by the number of employees in that gender subgroup to get an expected count. These counts are then summed to get the total `expected_qty` for the gift. Normalization to match total employees is **not** performed post-prediction.
5.  **API Output:** The final expected quantities and confidence scores are delivered via a RESTful API.

### API Flow (Conceptual):

1.  **Initial API Request:**
    ```json
    {
      "cvr": "12345678", // Company CVR for context
      "branch_no" : "B001",
      "gifts" : [
        {"id" : "P101", "description" : "Luxury Coffee Mug, Blue", "model_name": "CM-001", "vendor": "MugCo"},
        // ... other gifts
      ],
      "employees" : [
        {"name" : "Jane Doe"},
        {"name" : "John Smith"}
        // ... other employees
      ]
    }
    ```
2.  **Internal Data Reclassification & Enrichment:**
    *   Gifts are classified into attributes (e.g., `itemMainCategory`, `color`, `brand`).
    *   Employee names are processed to determine gender.
    *   Shop-specific features and historical context are resolved.
3.  **Prediction by ML Model (Selection Rates):** The model predicts a `selection_rate` for each gift-gender combination.
4.  **Scaling & Aggregation:** The predicted rates are scaled by employee counts per gender subgroup and then summed to get the `expected_qty` for each gift. No further normalization is applied.
5.  **API Response:**
    ```json
    [
      {"product_id" : "P101", "expected_qty" : 25, "confidence_score": 0.75},
      // ... other predictions (confidence_score typically between 0.5 and 0.95)
    ]
    ```

## User Experience Goals

### For Gavefabrikken Operations & Planning Teams:

-   **Accuracy & Reliability:** Provide demand forecasts that are significantly more accurate than current methods, leading to tangible improvements in inventory management.
-   **Actionability:** Deliver predictions in a clear, understandable format that can be directly used for inventory planning and procurement decisions.
-   **Timeliness:** Offer predictions quickly enough to be relevant for pre-season planning and in-season adjustments.
-   **Transparency (Future Goal):** Ideally, provide some insight into the drivers of predictions to build trust and aid in decision-making (though this is a secondary goal to accuracy).

### For Business Impact:

-   **Reduced Inventory Costs:** Minimize expenses related to overstocking, stockouts, and returns.
-   **Improved Operational Efficiency:** Streamline post-season operations by reducing the complexity of managing inventory imbalances.
-   **Enhanced Customer (Client & Employee) Satisfaction:** Ensure better availability of desired gifts, leading to a more positive experience for employees and greater satisfaction for Gavefabrikken's B2B clients.
-   **Data-Driven Decision Making:** Shift Gavefabrikken from a reactive to a proactive, analytics-driven approach for seasonal planning and inventory management.

## Transformation Goal

The ultimate goal is to transform Gavefabrikken's inventory management from a reactive, estimation-based process into a proactive, data-driven, and efficient operation. This system should enable Gavefabrikken to scale its business growth while maintaining high levels of customer satisfaction and operational efficiency.