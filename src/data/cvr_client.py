"""
CVR API client for fetching Danish company information.
"""

import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def fetch_industry_code(cvr: str) -> Optional[str]:
    """
    Fetch industry code from Danish CVR API.
    
    Args:
        cvr: Danish CVR number (8 digits)
        
    Returns:
        Industry code as string, or None if not found
    """
    url = f"https://cvrapi.dk/api?search={cvr}&country=dk"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                industry_code = data.get("industrycode")
                if industry_code:
                    logger.info(f"Found industry code {industry_code} for CVR {cvr}")
                    return str(industry_code)
                else:
                    logger.warning(f"No industry code in response for CVR {cvr}")
                    return None
            else:
                logger.error(f"CVR API returned status {response.status_code} for CVR {cvr}")
                return None
                
    except httpx.TimeoutException:
        logger.error(f"Timeout fetching industry code for CVR {cvr}")
        return None
    except Exception as e:
        logger.error(f"Error fetching industry code for CVR {cvr}: {e}")
        return None


async def get_company_info(cvr: str) -> Optional[dict]:
    """
    Fetch full company information from Danish CVR API.
    
    Args:
        cvr: Danish CVR number (8 digits)
        
    Returns:
        Company information dict, or None if not found
    """
    url = f"https://cvrapi.dk/api?search={cvr}&country=dk"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched company info for CVR {cvr}")
                return data
            else:
                logger.error(f"CVR API returned status {response.status_code} for CVR {cvr}")
                return None
                
    except httpx.TimeoutException:
        logger.error(f"Timeout fetching company info for CVR {cvr}")
        return None
    except Exception as e:
        logger.error(f"Error fetching company info for CVR {cvr}: {e}")
        return None