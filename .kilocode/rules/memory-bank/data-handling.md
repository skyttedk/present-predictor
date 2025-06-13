# Data Handling Guide

## Overview
This document provides detailed information about the actual data structures, field mappings, and processing requirements based on real data files.

## Historical Training Data Structure

### File: `src/data/historical/present.selection.historic.csv`

**Columns**:
1. `employee_shop` - Shop identifier (e.g., "2960")
2. `employee_branch` - Branch identifier (e.g., "621000", "841100")
3. `employee_gender` - Employee gender ("male", "female")
4. `product_main_category` - Primary product category
5. `product_sub_category` - Specific product subcategory  
6. `product_brand` - Brand name
7. `product_color` - Product color (mostly "NONE")
8. `product_durability` - Durability type ("durable", "consumable")
9. `product_target_gender` - Target gender ("unisex", "male", "female")
10. `product_utility_type` - Utility purpose ("practical", "aesthetic", "exclusive")
11. `product_type` - Usage type ("individual", "shareable")

### Sample Categories Observed
**Main Categories**:
- "Bags"
- "Home & Kitchen" 
- "Home & Decor"
- "Travel"
- "Tools & DIY"
- "Kitchen Appliance"

**Sub Categories**:
- "Toiletry Bag"
- "Cookware" 
- "knife set"
- "decorative flag"
- "lighting - lamp"
- "Hotel Stay"
- "Leaf Blower"
- "Blender"

**Brands**:
- "Markberg"
- "Fiskars"
- "Vex� Pro"
- "Kay Bojesen"
- "Tisvilde"
- "Comwell"
- "Gardenform"
- "Kitchen Master"

**Utility Types**:
- "practical" - functional items
- "aesthetic" - decorative items
- "exclusive" - premium experiences

**Durability**:
- "durable" - long-lasting items
- "consumable" - single-use items (e.g., hotel stays)

## API Classification Schema

### File: `src/data/product.attributes.schema.json`

**Schema Structure**:
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

**Enum Values**:
- `targetDemographic`: ["male", "female", "unisex"]
- `utilityType`: ["practical", "work", "aesthetic", "status", "sentimental", "exclusive"]
- `durability`: ["consumable", "durable"]
- `usageType`: ["shareable", "individual"]

## Field Mapping Requirements

### API Schema → Historical Data Columns
```
itemMainCategory    → product_main_category
itemSubCategory     → product_sub_category  
color              → product_color
brand              → product_brand
targetDemographic  → product_target_gender
utilityType        → product_utility_type
durability         → product_durability
usageType          → product_type
```

### Fields Not in Historical Data
- `vendor` - Extra field in API schema
- `valuePrice` - Extra field in API schema

### Fields Not in API Schema
- `employee_shop` - From API request branch_no
- `employee_branch` - From API request branch_no  
- `employee_gender` - Derived from employee names

## Data Processing Pipeline

### Step 1: API Request Processing
**Input Structure**:
```json
{
  "branch_no": "621000",
  "gifts": [
    {
      "product_id": "ABC123",
      "description": "Red ceramic mug with handle"
    }
  ],
  "employees": [
    {"name": "John Doe"},
    {"name": "Jane Smith"}
  ]
}
```

### Step 2: Classification Processing
**Required Operations**:
1. **Gift Description → Schema Attributes (OpenAI Assistant)**
   - **Use OpenAI Assistant API** (Assistant ID: `asst_BuFvA6iXF4xSyQ4px7Q5zjiN`)
   - Send product description to trained OpenAI agent via Assistant API
   - Receive structured JSON response matching `product.attributes.schema.json`
   - API flow: CreateThread → AddMessage → Run → GetRunStatus → GetThreadMessage
   - Apply JSON schema validation to response
   - Handle missing/default values

2. **Employee Name → Gender Classification (Enhanced gender_guesser)**
   - **Use enhanced gender_guesser with Danish name support**
   - Normalize names to proper case (handle hyphens, spaces, compound names)
   - Check against enhanced Danish names dictionary (ea, my, freja, bo, aksel, etc.)
   - Fallback to gender_guesser with Denmark country code
   - Handle uncertain results gracefully (return 'unknown' for low confidence)
   - Support for Nordic/European name patterns

3. **Branch Processing**
   - Map branch_no to employee_shop and employee_branch
   - Maintain consistent formatting

### Step 3: Historical Data Format Conversion
**Output Structure** (matching CSV columns):
```python
{
    'employee_shop': '2960',
    'employee_branch': '621000', 
    'employee_gender': 'male',
    'product_main_category': 'Home & Kitchen',
    'product_sub_category': 'Cookware',
    'product_brand': 'Fiskars',
    'product_color': 'NONE',
    'product_durability': 'durable',
    'product_target_gender': 'unisex',
    'product_utility_type': 'practical', 
    'product_type': 'individual'
}
```

## Data Quality Considerations

### Color Handling
- Most historical records have color as "NONE"
- When color is not applicable, use "NONE" as default
- Maintain consistency with historical patterns

### Brand Handling  
- Extract brand from descriptions when possible
- Use "NONE" when brand cannot be determined
- Validate against known brand patterns

### Category Consistency
- Maintain consistent category naming conventions
- Handle variations in capitalization and formatting
- Map similar categories to standard names

### Missing Value Strategy
- Use "NONE" for missing string fields (consistent with historical data)
- Use "unisex" as default for target demographics when unclear
- Use "individual" as default for usage type when unclear

## Implementation Notes

### Classification Accuracy Requirements
- Gift description parsing must achieve >85% accuracy
- Name-to-gender classification should handle Nordic/European names
- Category mapping should be robust to description variations

### Performance Considerations
- Cache classification results for repeated descriptions
- Batch process multiple gifts efficiently
- Validate schema compliance before ML processing

### Error Handling
- Gracefully handle unparseable descriptions
- Provide fallback values for missing classifications
- Log classification confidence scores for monitoring

## Testing Strategy

### Unit Tests Required
- Test classification with sample descriptions
- Validate field mapping completeness
- Test edge cases and error conditions

### Integration Tests
- End-to-end pipeline with real data samples
- Validate output format matches historical structure
- Test performance with realistic batch sizes

### Data Validation
- Compare classified data distributions with historical patterns
- Monitor classification accuracy over time
- Validate enum value compliance