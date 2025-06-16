from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import Dict

from src.database.users import authenticate_user

app = FastAPI(
    title="Predictive Gift Selection API",
    description="API for predicting gift selection quantities.",
    version="0.1.0",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_current_user(api_key: str = Depends(api_key_header)) -> Dict:
    """
    Dependency to authenticate user via API key.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: API key is missing",
            headers={"WWW-Authenticate": "Header"},
        )
    user = authenticate_user(api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Header"},
        )
    return user

@app.get("/test")
async def test_endpoint(current_user: Dict = Depends(get_current_user)):
    """
    A simple test endpoint, protected by API key.
    """
    return {"message": f"Test endpoint is working! Hello {current_user.get('username')}!"}