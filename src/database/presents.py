"""
Present classification caching functions.
"""
import hashlib
from typing import Dict, Optional, List, Any
import logging

from . import db_factory

logger = logging.getLogger(__name__)

def get_present_hash(present_name: str, model_name: str, model_no: str) -> str:
    """
    Generate hash for present details.

    Args:
        present_name: Present name text
        model_name: Model name text
        model_no: Model number text

    Returns:
        MD5 hash of the combined present details
    """
    combined_details = f"{present_name}_{model_name}_{model_no}".lower().strip()
    return hashlib.md5(combined_details.encode()).hexdigest()

def get_cached_present_classification(
    present_name: str,
    model_name: str,
    model_no: str
) -> Optional[Dict[str, Any]]:
    """
    Check if present classification exists in cache using present_hash.

    Args:
        present_name: Present name
        model_name: Model name
        model_no: Model number

    Returns:
        Cached classification if found, None otherwise
    """
    present_hash = get_present_hash(present_name, model_name, model_no)
    query = """
        SELECT * FROM present_attributes
        WHERE present_hash = ?
        AND classification_status = 'success'
        ORDER BY classified_at DESC
        LIMIT 1
    """
    results = db_factory.execute_query(query, (present_hash,))

    if results:
        logger.debug(f"Cache hit for present_hash: {present_hash}")
        return results[0]

    logger.debug(f"Cache miss for present_hash: {present_hash}")
    return None

def cache_present_classification(
    present_name: str,
    model_name: str,
    model_no: str,
    present_vendor: Optional[str],
    thread_id: str,
    run_id: str,
    attributes: Dict[str, Any],
    status: str = 'success'
) -> int:
    """
    Cache present classification from OpenAI.

    Args:
        present_name: Present name
        model_name: Model name
        model_no: Model number
        present_vendor: Vendor of the present itself
        thread_id: OpenAI thread ID
        run_id: OpenAI run ID
        attributes: Classification attributes dictionary (contains classified 'vendor')
        status: Classification status ('success', 'failed', 'partial')

    Returns:
        ID of the cached entry
    """
    present_hash = get_present_hash(present_name, model_name, model_no)

    query = """
        INSERT OR REPLACE INTO present_attributes
        (present_hash, present_name, present_vendor, model_name, model_no,
         thread_id, run_id,
         item_main_category, item_sub_category, color, brand, vendor,
         value_price, target_demographic, utility_type, durability, usage_type,
         classification_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = (
        present_hash,
        present_name,
        present_vendor,
        model_name,
        model_no,
        thread_id,
        run_id,
        attributes.get("itemMainCategory"),
        attributes.get("itemSubCategory"),
        attributes.get("color"),
        attributes.get("brand"),
        attributes.get("vendor"), # This is the classified vendor from attributes
        attributes.get("valuePrice"),
        attributes.get("targetDemographic"),
        attributes.get("utilityType"),
        attributes.get("durability"),
        attributes.get("usageType"),
        status
    )
    
    cache_id = db_factory.execute_write(query, params)

    logger.info(f"Cached classification for present_hash: {present_hash}")
    return cache_id

def batch_cache_present_classifications(
    classifications: List[Dict[str, Any]]
) -> int:
    """
    Cache multiple present classifications at once.

    Args:
        classifications: List of classification dictionaries. Each dict should contain
                         'present_name', 'model_name', 'model_no', 'present_vendor',
                         'attributes' (containing classified 'vendor'),
                         and optionally 'thread_id', 'run_id', 'status'.
    Returns:
        Number of cached entries
    """
    query = """
        INSERT OR REPLACE INTO present_attributes
        (present_hash, present_name, present_vendor, model_name, model_no,
         thread_id, run_id,
         item_main_category, item_sub_category, color, brand, vendor,
         value_price, target_demographic, utility_type, durability, usage_type,
         classification_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params_list = []
    for item in classifications:
        present_hash = get_present_hash(item['present_name'], item['model_name'], item['model_no'])
        params = (
            present_hash,
            item['present_name'],
            item.get('present_vendor'),
            item['model_name'],
            item['model_no'],
            item.get('thread_id'),
            item.get('run_id'),
            item['attributes'].get("itemMainCategory"),
            item['attributes'].get("itemSubCategory"),
            item['attributes'].get("color"),
            item['attributes'].get("brand"),
            item['attributes'].get("vendor"), # Classified vendor from attributes
            item['attributes'].get("valuePrice"),
            item['attributes'].get("targetDemographic"),
            item['attributes'].get("utilityType"),
            item['attributes'].get("durability"),
            item['attributes'].get("usageType"),
            item.get('status', 'success')
        )
        params_list.append(params)

    count = db_factory.execute_many(query, params_list)
    logger.info(f"Batch cached {count} present classifications")
    return count

def update_classified_present_attributes(
    present_hash: str,
    attributes: Dict[str, Any],
    new_status: str,
    thread_id: Optional[str] = None,
    run_id: Optional[str] = None
) -> int:
    """
    Updates the attributes and status of an already existing present classification entry.
    Uses specific fields from the attributes dictionary.
    """
    fields_to_set = []
    params_list = []

    attribute_map = {
        "itemMainCategory": "item_main_category",
        "itemSubCategory": "item_sub_category",
        "color": "color",
        "brand": "brand",
        "vendor": "vendor", 
        "valuePrice": "value_price",
        "targetDemographic": "target_demographic",
        "utilityType": "utility_type",
        "durability": "durability",
        "usageType": "usage_type",
    }

    for attr_key, db_col in attribute_map.items():
        if attr_key in attributes and attributes[attr_key] is not None:
            fields_to_set.append(f"{db_col} = ?")
            params_list.append(attributes[attr_key])

    fields_to_set.append("classification_status = ?")
    params_list.append(new_status)
    fields_to_set.append("classified_at = datetime('now')") # Update timestamp

    if thread_id:
        fields_to_set.append("thread_id = ?")
        params_list.append(thread_id)
    if run_id:
        fields_to_set.append("run_id = ?")
        params_list.append(run_id)
    
    if not fields_to_set: # Should not happen if at least status is being updated
        logger.warning(f"No fields to update for present_hash: {present_hash} (this is unexpected).")
        return 0

    query = f"""
        UPDATE present_attributes
        SET {', '.join(fields_to_set)}
        WHERE present_hash = ?
    """
    params_list.append(present_hash)
    
    updated_rows = db_factory.execute_write(query, tuple(params_list))
    if updated_rows > 0:
        logger.info(f"Successfully updated attributes for present_hash: {present_hash} to status '{new_status}'")
    else:
        logger.warning(f"Failed to update or no record found for present_hash: {present_hash} with status '{new_status}'")
    return updated_rows

def get_pending_classification_presents() -> List[Dict[str, Any]]:
    """
    Retrieves all presents that are pending classification.

    Returns:
        List of dictionaries, each representing a present pending classification.
        Includes present_hash, present_name, present_vendor, model_name, model_no.
    """
    query = """
        SELECT present_hash, present_name, present_vendor, model_name, model_no
        FROM present_attributes
        WHERE classification_status = 'pending_classification'
    """
    results = db_factory.execute_query(query)
    if results:
        logger.info(f"Found {len(results)} presents pending classification.")
    else:
        logger.info("No presents found pending classification.")
    return results
def get_present_classification_stats() -> Dict[str, Any]:
    """
    Get statistics about the present classification cache.

    Returns:
        Dictionary with cache statistics
    """
    query = """
        SELECT
            COUNT(*) as total_cached,
            COUNT(DISTINCT present_hash) as unique_presents,
            COUNT(CASE WHEN classification_status = 'success' THEN 1 END) as successful,
            COUNT(CASE WHEN classification_status = 'failed' THEN 1 END) as failed,
            MIN(classified_at) as oldest_entry,
            MAX(classified_at) as newest_entry
        FROM present_attributes
    """

    stats_list = db_factory.execute_query(query)
    stats = stats_list[0] if stats_list else {}


    # Get distribution by category
    category_query = """
        SELECT
            item_main_category,
            COUNT(*) as count
        FROM present_attributes
        WHERE classification_status = 'success'
        AND item_main_category IS NOT NULL
        GROUP BY item_main_category
        ORDER BY count DESC
        LIMIT 10
    """

    categories = db_factory.execute_query(category_query)

    return {
        **stats,
        "top_categories": categories
    }

def cleanup_old_present_cache(days: int = 30) -> int:
    """
    Remove old cached present classifications.

    Args:
        days: Number of days to keep cache entries

    Returns:
        Number of deleted entries
    """
    query = """
        DELETE FROM present_attributes
        WHERE classified_at < datetime('now', '-' || ? || ' days')
    """

    deleted = db_factory.execute_write(query, (days,))
    logger.info(f"Cleaned up {deleted} old present cache entries")

    return deleted

def invalidate_present_cache(present_name: str, model_name: str, model_no: str) -> bool:
    """
    Invalidate cache for a specific present by its details.

    Args:
        present_name: Present name
        model_name: Model name
        model_no: Model number

    Returns:
        True if cache was invalidated, False if present not found or already invalidated.
    """
    present_hash = get_present_hash(present_name, model_name, model_no)
    query = """
        UPDATE present_attributes
        SET classification_status = 'invalidated'
        WHERE present_hash = ? AND classification_status != 'invalidated'
    """
    affected = db_factory.execute_write(query, (present_hash,))

    if affected > 0:
        logger.info(f"Invalidated cache for present_hash: {present_hash}")
        return True
    logger.info(f"No active cache entry found to invalidate for present_hash: {present_hash}")
    return False