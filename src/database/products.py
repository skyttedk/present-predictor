"""
Product classification caching functions.
"""
import hashlib
from typing import Dict, Optional, List, Any
import logging

from .db import execute_query, execute_write, execute_many

logger = logging.getLogger(__name__)

def get_product_hash(description: str) -> str:
    """
    Generate hash for product description.

    Args:
        description: Product description text

    Returns:
        MD5 hash of the description
    """
    return hashlib.md5(description.lower().strip().encode()).hexdigest()

def get_cached_classification(
    product_id: str,
    description: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Check if product classification exists in cache.

    Args:
        product_id: Product identifier
        description: Product description (optional)

    Returns:
        Cached classification if found, None otherwise
    """
    if description:
        product_hash = get_product_hash(description)
        query = """
            SELECT * FROM product_attributes
            WHERE (product_id = ? OR product_hash = ?)
            AND classification_status = 'success'
            ORDER BY classified_at DESC
            LIMIT 1
        """
        results = execute_query(query, (product_id, product_hash))
    else:
        query = """
            SELECT * FROM product_attributes
            WHERE product_id = ?
            AND classification_status = 'success'
            ORDER BY classified_at DESC
            LIMIT 1
        """
        results = execute_query(query, (product_id,))

    if results:
        logger.debug(f"Cache hit for product: {product_id}")
        return results[0]

    logger.debug(f"Cache miss for product: {product_id}")
    return None

def cache_classification(
    product_id: str,
    description: str,
    thread_id: str,
    run_id: str,
    attributes: Dict[str, Any],
    status: str = 'success'
) -> int:
    """
    Cache product classification from OpenAI.

    Args:
        product_id: Product identifier
        description: Product description
        thread_id: OpenAI thread ID
        run_id: OpenAI run ID
        attributes: Classification attributes dictionary
        status: Classification status ('success', 'failed', 'partial')

    Returns:
        ID of the cached entry
    """
    product_hash = get_product_hash(description)

    query = """
        INSERT OR REPLACE INTO product_attributes
        (product_id, product_hash, thread_id, run_id,
         item_main_category, item_sub_category, color, brand, vendor,
         value_price, target_demographic, utility_type, durability, usage_type,
         raw_description, classification_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cache_id = execute_write(query, (
        product_id,
        product_hash,
        thread_id,
        run_id,
        attributes.get("itemMainCategory"),
        attributes.get("itemSubCategory"),
        attributes.get("color"),
        attributes.get("brand"),
        attributes.get("vendor"),
        attributes.get("valuePrice"),
        attributes.get("targetDemographic"),
        attributes.get("utilityType"),
        attributes.get("durability"),
        attributes.get("usageType"),
        description,
        status
    ))

    logger.info(f"Cached classification for product: {product_id}")
    return cache_id

def batch_cache_classifications(
    classifications: List[Dict[str, Any]]
) -> int:
    """
    Cache multiple product classifications at once.

    Args:
        classifications: List of classification dictionaries

    Returns:
        Number of cached entries
    """
    query = """
        INSERT OR REPLACE INTO product_attributes
        (product_id, product_hash, thread_id, run_id,
         item_main_category, item_sub_category, color, brand, vendor,
         value_price, target_demographic, utility_type, durability, usage_type,
         raw_description, classification_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params_list = []
    for item in classifications:
        product_hash = get_product_hash(item['description'])
        params = (
            item['product_id'],
            product_hash,
            item.get('thread_id'),
            item.get('run_id'),
            item['attributes'].get("itemMainCategory"),
            item['attributes'].get("itemSubCategory"),
            item['attributes'].get("color"),
            item['attributes'].get("brand"),
            item['attributes'].get("vendor"),
            item['attributes'].get("valuePrice"),
            item['attributes'].get("targetDemographic"),
            item['attributes'].get("utilityType"),
            item['attributes'].get("durability"),
            item['attributes'].get("usageType"),
            item['description'],
            item.get('status', 'success')
        )
        params_list.append(params)

    count = execute_many(query, params_list)
    logger.info(f"Batch cached {count} product classifications")
    return count

def get_classification_stats() -> Dict[str, Any]:
    """
    Get statistics about the classification cache.

    Returns:
        Dictionary with cache statistics
    """
    query = """
        SELECT
            COUNT(*) as total_cached,
            COUNT(DISTINCT product_id) as unique_products,
            COUNT(CASE WHEN classification_status = 'success' THEN 1 END) as successful,
            COUNT(CASE WHEN classification_status = 'failed' THEN 1 END) as failed,
            MIN(classified_at) as oldest_entry,
            MAX(classified_at) as newest_entry
        FROM product_attributes
    """

    stats_list = execute_query(query)
    stats = stats_list[0] if stats_list else {}


    # Get distribution by category
    category_query = """
        SELECT
            item_main_category,
            COUNT(*) as count
        FROM product_attributes
        WHERE classification_status = 'success'
        AND item_main_category IS NOT NULL
        GROUP BY item_main_category
        ORDER BY count DESC
        LIMIT 10
    """

    categories = execute_query(category_query)

    return {
        **stats,
        "top_categories": categories
    }

def cleanup_old_cache(days: int = 30) -> int:
    """
    Remove old cached classifications.

    Args:
        days: Number of days to keep cache entries

    Returns:
        Number of deleted entries
    """
    query = """
        DELETE FROM product_attributes
        WHERE classified_at < datetime('now', '-' || ? || ' days')
    """

    deleted = execute_write(query, (days,))
    logger.info(f"Cleaned up {deleted} old cache entries")

    return deleted

def invalidate_product_cache(product_id: str) -> bool:
    """
    Invalidate cache for a specific product.

    Args:
        product_id: Product identifier

    Returns:
        True if cache was invalidated, False if product not found
    """
    query = """
        UPDATE product_attributes
        SET classification_status = 'invalidated'
        WHERE product_id = ?
    """

    affected = execute_write(query, (product_id,))

    if affected > 0:
        logger.info(f"Invalidated cache for product: {product_id}")
        return True
    return False