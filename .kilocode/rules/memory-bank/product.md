# Product Vision

## Why This Project Exists

Gavefabrikken operates a B2B gift distribution service where companies provide curated gift selections to their employees through dedicated online portals. The company faces significant challenges in demand forecasting, leading to inventory imbalances, operational overhead, and customer dissatisfaction.

## Problems We Solve

### Primary Pain Points
- **Inventory Imbalances**: Current manual estimation methods result in overstocking and stockouts
- **Operational Overhead**: Significant post-season complexities from poor demand prediction
- **Financial Impact**: Increased costs from back-orders and surplus gift returns
- **Customer Satisfaction**: Unavailable gift selections due to poor inventory planning

### Current State vs Desired State
**Current**: Manual estimation + basic statistical averages
**Target**: ML-powered demand prediction system with accurate quantity forecasts

## How It Should Work

### Core Workflow
1. **Data Input**: Companies submit gift catalogs and employee information
2. **Data Processing**: System classifies gifts and employee demographics
3. **Prediction**: ML model generates demand forecasts per product
4. **Output**: Actionable quantity recommendations for inventory planning

### API Flow
```
Step 1: Initial Request (product_id, description, employee names)
↓
Step 2: Internal Reclassification (categorized gifts, gender demographics)
↓
Step 3: Prediction Response (product_id, expected_qty)
```

## User Experience Goals

### For Gavefabrikken Operations Team
- **Simplicity**: Single API call to get demand forecasts
- **Accuracy**: Reliable predictions to reduce inventory risks
- **Speed**: Real-time predictions for operational planning
- **Transparency**: Clear understanding of prediction rationale

### For Business Impact
- Reduced inventory management costs
- Minimized post-season operational complexity
- Enhanced customer satisfaction through better availability
- Data-driven decision making for seasonal planning

## Success Metrics

### Technical Metrics
- Prediction accuracy (target: >85% within ±20% of actual demand)
- API response time (target: <2 seconds)
- System uptime (target: 99.5%)

### Business Metrics
- Reduction in inventory imbalances (target: 40% improvement)
- Decrease in post-season operational costs
- Improved customer satisfaction scores
- ROI from reduced surplus and back-order costs

## Transformation Goal

Transform Gavefabrikken's reactive inventory approach into a proactive, analytics-driven system capable of scaling with business growth while maintaining operational efficiency.