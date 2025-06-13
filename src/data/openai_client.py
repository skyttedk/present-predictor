"""
OpenAI Assistant API client for gift classification.
Based on Business Central implementation pattern.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from ..config.settings import get_settings
from .schemas.data_models import GiftAttributes

logger = logging.getLogger(__name__)


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI API."""
    api_key: str
    assistant_id: str = "asst_BuFvA6iXF4xSyQ4px7Q5zjiN"
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 30
    max_retries: int = 3


class OpenAIClassificationError(Exception):
    """Custom exception for OpenAI classification errors."""
    pass


class OpenAIAssistantClient:
    """
    OpenAI Assistant API client for product classification.
    Implements the same pattern as the Business Central integration.
    """
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        if config is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            config = OpenAIConfig(api_key=api_key)
        
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "OpenAI-Beta": "assistants=v2",
                "Content-Type": "application/json"
            }
        )
    
    async def classify_product(self, description: str) -> GiftAttributes:
        """
        Classify a product description using OpenAI Assistant.
        
        Args:
            description: Product description to classify
            
        Returns:
            GiftAttributes: Classified product attributes
            
        Raises:
            OpenAIClassificationError: If classification fails
        """
        if not description or not description.strip():
            raise OpenAIClassificationError("Product description cannot be empty")
        
        try:
            # Step 1: Create thread
            thread_id = await self._create_thread()
            
            # Step 2: Add message
            await self._add_message(thread_id, description.strip())
            
            # Step 3: Run assistant
            run_id = await self._create_run(thread_id)
            
            # Step 4: Wait for completion
            await self._wait_for_completion(thread_id, run_id)
            
            # Step 5: Get response
            response_data = await self._get_thread_messages(thread_id)
            
            # Step 6: Parse and validate response
            return self._parse_classification_response(response_data)
            
        except Exception as e:
            logger.error(f"Product classification failed for '{description}': {e}")
            raise OpenAIClassificationError(f"Classification failed: {e}")
    
    async def _create_thread(self) -> str:
        """Create a new conversation thread."""
        url = f"{self.config.base_url}/threads"
        
        async with self.client as client:
            response = await client.post(url, json={})
            response.raise_for_status()
            data = response.json()
            return data["id"]
    
    async def _add_message(self, thread_id: str, message: str) -> str:
        """Add a message to the thread."""
        url = f"{self.config.base_url}/threads/{thread_id}/messages"
        payload = {
            "role": "user",
            "content": message
        }
        
        async with self.client as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["id"]
    
    async def _create_run(self, thread_id: str) -> str:
        """Create a run for the thread."""
        url = f"{self.config.base_url}/threads/{thread_id}/runs"
        payload = {
            "assistant_id": self.config.assistant_id
        }
        
        async with self.client as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["id"]
    
    async def _wait_for_completion(self, thread_id: str, run_id: str, max_wait: int = 60) -> None:
        """Wait for the run to complete."""
        url = f"{self.config.base_url}/threads/{thread_id}/runs/{run_id}"
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            async with self.client as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                status = data.get("status")
                logger.debug(f"Run status: {status}")
                
                if status == "completed":
                    return
                elif status in ["failed", "cancelled", "expired"]:
                    error_msg = data.get("last_error", {}).get("message", "Unknown error")
                    raise OpenAIClassificationError(f"Run {status}: {error_msg}")
                elif status in ["queued", "in_progress", "requires_action"]:
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.warning(f"Unknown run status: {status}")
                    await asyncio.sleep(1)
        
        raise OpenAIClassificationError(f"Run timed out after {max_wait} seconds")
    
    async def _get_thread_messages(self, thread_id: str) -> Dict[str, Any]:
        """Get the latest message from the thread."""
        url = f"{self.config.base_url}/threads/{thread_id}/messages"
        
        async with self.client as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("data"):
                raise OpenAIClassificationError("No messages found in thread")
            
            # Get the first (latest) message
            latest_message = data["data"][0]
            content = latest_message.get("content", [])
            
            if not content:
                raise OpenAIClassificationError("No content in latest message")
            
            # Extract text content
            text_content = content[0].get("text", {}).get("value", "")
            if not text_content:
                raise OpenAIClassificationError("No text content in message")
            
            return {"content": text_content}
    
    def _parse_classification_response(self, response_data: Dict[str, Any]) -> GiftAttributes:
        """
        Parse the classification response from OpenAI.
        
        Args:
            response_data: Response data from OpenAI
            
        Returns:
            GiftAttributes: Parsed and validated attributes
        """
        content = response_data.get("content", "")
        
        try:
            # Clean and parse JSON
            cleaned_content = self._clean_json_text(content)
            classification_data = json.loads(cleaned_content)
            
            # Validate and create GiftAttributes
            return GiftAttributes(**classification_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {content}")
            raise OpenAIClassificationError(f"Invalid JSON response: {e}")
        except ValidationError as e:
            logger.error(f"Failed to validate classification data: {e}")
            raise OpenAIClassificationError(f"Invalid classification data: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            raise OpenAIClassificationError(f"Parsing error: {e}")
    
    def _clean_json_text(self, json_text: str) -> str:
        """
        Clean JSON text response.
        Based on the Business Central ClearJsonText procedure.
        """
        if not json_text:
            return "{}"
        
        cleaned = json_text.strip()
        
        # Remove escape characters and whitespace
        cleaned = cleaned.replace('\\"', '"')
        cleaned = cleaned.replace('\\n', '')
        cleaned = cleaned.replace('\\r', '')
        cleaned = cleaned.replace('\\t', '')
        cleaned = cleaned.replace('\n', '')
        cleaned = cleaned.replace('\r', '')
        cleaned = cleaned.replace('\t', '')
        
        # Remove surrounding quotes if present
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        
        # Handle markdown code blocks
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        
        return cleaned.strip()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Add missing import
import asyncio


def create_openai_client() -> OpenAIAssistantClient:
    """Create an OpenAI Assistant client with default configuration."""
    return OpenAIAssistantClient()


async def classify_product_description(description: str) -> GiftAttributes:
    """
    Convenience function to classify a single product description.
    
    Args:
        description: Product description to classify
        
    Returns:
        GiftAttributes: Classified attributes
    """
    async with create_openai_client() as client:
        return await client.classify_product(description)