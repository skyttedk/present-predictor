# Postman Usage Guide - Predict Presents API

This guide provides step-by-step instructions for testing the Predict Presents API using Postman.

## API Base Information

- **Base URL**: `https://predict-presents-api-b58dda526ddb.herokuapp.com`
- **Authentication**: All endpoints require `X-API-Key` header
- **Test API Key**: `zfzYvn2FmWixrrDsbtw6dUUjgg7sgbi-ce_YFjKKPMU`

## Setting up Postman Collection

### 1. Create Environment Variables

First, set up environment variables in Postman for easier testing:

1. Click the gear icon (⚙️) in the top right
2. Select "Manage Environments"
3. Click "Add" to create a new environment
4. Name it "Predict Presents API"
5. Add these variables:

| Variable | Value |
|----------|-------|
| `base_url` | `https://predict-presents-api-b58dda526ddb.herokuapp.com` |
| `api_key` | `zfzYvn2FmWixrrDsbtw6dUUjgg7sgbi-ce_YFjKKPMU` |

### 2. Set Authorization Header

For each request, add the authorization header:
- **Key**: `X-API-Key`
- **Value**: `{{api_key}}`

## API Endpoints

### 1. Test Endpoint

**Purpose**: Verify API connectivity and authentication

**Request Details**:
- **Method**: `GET`
- **URL**: `{{base_url}}/test`
- **Headers**: 
  - `X-API-Key: {{api_key}}`

**Expected Response** (200 OK):
```json
{
  "message": "API is working! Authentication successful.",
  "user": "testuser",
  "timestamp": "2025-06-17T08:50:00.000Z"
}
```

---

### 2. Add Present

**Purpose**: Add a new present for classification

**Request Details**:
- **Method**: `POST`
- **URL**: `{{base_url}}/addPresent`
- **Headers**: 
  - `X-API-Key: {{api_key}}`
  - `Content-Type: application/json`

**Request Body** (JSON):
```json
{
  "present_name": "Smart Watch",
  "present_vendor": "TechCorp",
  "model_name": "Sport Edition",
  "model_no": "SW-2024"
}
```

**Expected Response** (201 Created):
```json
{
  "message": "Present added successfully",
  "present_hash": "abc123def456...",
  "status": "pending_classification",
  "processing_time_ms": 45.2
}
```

**Alternative Response** (200 OK - Present already exists):
```json
{
  "message": "Present already exists",
  "present_hash": "abc123def456...",
  "status": "success",
  "processing_time_ms": 12.1
}
```

---

### 3. Count Presents

**Purpose**: Get statistics on presents by classification status

**Request Details**:
- **Method**: `GET`
- **URL**: `{{base_url}}/countPresents`
- **Headers**: 
  - `X-API-Key: {{api_key}}`

**Expected Response** (200 OK):
```json
{
  "message": "Present counts retrieved successfully",
  "total_presents": 5,
  "status_counts": [
    {
      "status": "success",
      "count": 3
    },
    {
      "status": "pending_classification",
      "count": 1
    },
    {
      "status": "error_openai_api",
      "count": 1
    }
  ],
  "processing_time_ms": 29.7
}
```

---

### 4. CSV Import (Bulk)

**Purpose**: Import multiple presents from CSV file

**Request Details**:
- **Method**: `POST`
- **URL**: `{{base_url}}/addPresentsProcessed`
- **Headers**: 
  - `X-API-Key: {{api_key}}`
- **Body Type**: `form-data`
- **File Parameter**: `file` (select your CSV file)

**CSV Format Example**:
```csv
present_name,present_vendor,model_name,model_no,itemMainCategory,itemSubCategory,color,brand,vendor,valuePrice,targetDemographic,utilityType,durability,usageType
Coffee Mug,MugCorp,Ceramic Blue,CM-001,Home & Kitchen,Drinkware,Blue,MugCorp,MugCorp,15.99,unisex,practical,durable,individual
Wireless Headphones,AudioTech,Noise Cancelling,WH-200,Electronics,Audio,Black,AudioTech,AudioTech,89.99,unisex,practical,durable,individual
```

**Expected Response** (200 OK):
```json
{
  "message": "CSV import completed successfully",
  "total_records": 150,
  "imported": 120,
  "skipped": 25,
  "errors": 5,
  "processing_time_ms": 3245.8,
  "performance_note": "Optimized bulk processing (500x faster than individual inserts)"
}
```

---

### 5. Delete All Presents (Testing Only)

**Purpose**: Clear all presents from database (for testing)

**Request Details**:
- **Method**: `POST`
- **URL**: `{{base_url}}/deleteAllPresents`
- **Headers**: 
  - `X-API-Key: {{api_key}}`

**Expected Response** (200 OK):
```json
{
  "message": "All presents deleted successfully",
  "deleted_count": 15,
  "processing_time_ms": 125.3
}
```

## Testing Scenarios

### Scenario 1: Basic API Testing

1. **Test Connection**: Start with the `/test` endpoint to verify connectivity
2. **Add Present**: Use `/addPresent` to add a sample present
3. **Check Status**: Use `/countPresents` to see the present was added
4. **Verify Processing**: Wait 2 minutes and check `/countPresents` again to see if classification completed

### Scenario 2: CSV Import Testing

1. **Prepare CSV**: Create a CSV file with present data following the format above
2. **Import Data**: Use `/addPresentsProcessed` to bulk import
3. **Verify Import**: Use `/countPresents` to check imported count
4. **Monitor Processing**: Check `/countPresents` periodically to see classification progress

### Scenario 3: Error Handling

1. **Missing API Key**: Try any endpoint without the `X-API-Key` header
2. **Invalid API Key**: Use an incorrect API key value
3. **Invalid JSON**: Send malformed JSON to `/addPresent`
4. **Missing Fields**: Send incomplete data to `/addPresent`

## Common Postman Issues and Solutions

### Issue 1: Malformed URL

**Problem**: URL shows `http://https://...` (double protocol)

**Solution**: 
- Use only `https://predict-presents-api-b58dda526ddb.herokuapp.com`
- Don't include `http://` in the base_url variable

### Issue 2: 401 Unauthorized

**Problem**: Missing or incorrect API key

**Solution**:
- Ensure `X-API-Key` header is present
- Use the correct API key: `zfzYvn2FmWixrrDsbtw6dUUjgg7sgbi-ce_YFjKKPMU`
- Check for extra spaces or characters

### Issue 3: 405 Method Not Allowed

**Problem**: Using wrong HTTP method

**Solution**:
- `/test` and `/countPresents` are GET requests
- `/addPresent`, `/addPresentsProcessed`, `/deleteAllPresents` are POST requests

### Issue 4: CSV Upload Issues

**Problem**: CSV file not uploading correctly

**Solution**:
- Use `form-data` body type (not `raw` or `x-www-form-urlencoded`)
- Set parameter name to `file`
- Select file type as "File" (not "Text")

## Response Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | OK | Successful GET request |
| 201 | Created | Successfully created new resource |
| 400 | Bad Request | Invalid JSON or missing required fields |
| 401 | Unauthorized | Missing or invalid API key |
| 405 | Method Not Allowed | Wrong HTTP method for endpoint |
| 422 | Unprocessable Entity | Valid JSON but invalid data values |
| 500 | Internal Server Error | Server-side error |

## Background Processing

The API includes background classification tasks that run every 2 minutes:

1. **Present Classification**: Newly added presents are automatically classified using OpenAI
2. **Status Updates**: Use `/countPresents` to monitor classification progress
3. **Status Values**:
   - `pending_classification`: Waiting for OpenAI classification
   - `success`: Successfully classified
   - `error_openai_api`: Classification failed due to API error

## Performance Notes

- **CSV Import**: Optimized for bulk operations (500x faster than individual inserts)
- **Response Times**: Most endpoints respond within 50ms
- **Background Tasks**: Classification happens asynchronously, doesn't affect API response times
- **Database**: PostgreSQL with connection pooling for high performance

## Support and Troubleshooting

If you encounter issues:

1. **Check Heroku Logs**: Contact the development team for recent error logs
2. **Verify Environment**: Ensure you're using the production URL and valid API key
3. **Test Basic Connectivity**: Start with the `/test` endpoint
4. **Check Request Format**: Ensure headers and body format match the examples above

For additional support, contact the development team with:
- Request details (method, URL, headers, body)
- Expected vs actual response
- Error messages or status codes received