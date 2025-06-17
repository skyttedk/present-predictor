"""
API call logging functions.
"""
import json
from typing import Dict, List, Optional, Any
import logging

from . import db_factory

logger = logging.getLogger(__name__)

def log_api_call(
    user_id: int,
    api_route: str,
    request_payload: Dict[str, Any],
    response_payload: Dict[str, Any],
    status_code: int,
    response_time_ms: float,
    error_message: Optional[str] = None
) -> int:
    """
    Log an API call to the database.

    Args:
        user_id: ID of the user making the request
        api_route: API endpoint route
        request_payload: Request body as dictionary
        response_payload: Response body as dictionary
        status_code: HTTP status code
        response_time_ms: Response time in milliseconds
        error_message: Optional error message

    Returns:
        ID of the inserted log entry
    """
    query = """
        INSERT INTO user_api_call_log
        (user_id, api_route, request_payload, response_payload,
         response_status_code, response_time_ms, error_message)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    log_id = db_factory.execute_write(query, (
        user_id,
        api_route,
        json.dumps(request_payload),
        json.dumps(response_payload),
        status_code,
        response_time_ms,
        error_message
    ))

    if status_code >= 400:
        logger.warning(f"API error logged - User: {user_id}, Route: {api_route}, Status: {status_code}")

    return log_id

def get_user_api_stats(user_id: int, days: int = 30) -> Dict[str, Any]:
    """
    Get API usage statistics for a user.

    Args:
        user_id: User ID
        days: Number of days to look back (default: 30)

    Returns:
        Dictionary with usage statistics
    """
    query = """
        SELECT
            COUNT(*) as total_calls,
            AVG(response_time_ms) as avg_response_time,
            MAX(response_time_ms) as max_response_time,
            MIN(response_time_ms) as min_response_time,
            COUNT(CASE WHEN response_status_code >= 400 THEN 1 END) as error_count,
            COUNT(DISTINCT api_route) as unique_endpoints
        FROM user_api_call_log
        WHERE user_id = %s
        AND date_time >= CURRENT_TIMESTAMP - INTERVAL '%s days'
    """

    stats_list = db_factory.execute_query(query, (user_id, days))
    stats = stats_list[0] if stats_list else {}


    # Get calls by endpoint
    endpoint_query = """
        SELECT
            api_route,
            COUNT(*) as call_count,
            AVG(response_time_ms) as avg_response_time
        FROM user_api_call_log
        WHERE user_id = %s
        AND date_time >= CURRENT_TIMESTAMP - INTERVAL '%s days'
        GROUP BY api_route
        ORDER BY call_count DESC
    """

    endpoints = db_factory.execute_query(endpoint_query, (user_id, days))

    return {
        **stats,
        "endpoints": endpoints,
        "period_days": days
    }

def get_system_api_stats(hours: int = 24) -> Dict[str, Any]:
    """
    Get system-wide API statistics.

    Args:
        hours: Number of hours to look back (default: 24)

    Returns:
        Dictionary with system statistics
    """
    query = """
        SELECT
            COUNT(*) as total_calls,
            COUNT(DISTINCT user_id) as active_users,
            AVG(response_time_ms) as avg_response_time,
            COUNT(CASE WHEN response_status_code >= 400 THEN 1 END) as error_count,
            COUNT(CASE WHEN response_status_code >= 500 THEN 1 END) as server_error_count
        FROM user_api_call_log
        WHERE date_time >= CURRENT_TIMESTAMP - INTERVAL '%s hours'
    """

    stats_list = db_factory.execute_query(query, (hours,))
    stats = stats_list[0] if stats_list else {}

    # Get top users
    user_query = """
        SELECT
            u.username,
            COUNT(*) as call_count
        FROM user_api_call_log l
        JOIN "user" u ON l.user_id = u.id
        WHERE l.date_time >= CURRENT_TIMESTAMP - INTERVAL '%s hours'
        GROUP BY l.user_id, u.username
        ORDER BY call_count DESC
        LIMIT 10
    """

    top_users = db_factory.execute_query(user_query, (hours,))

    return {
        **stats,
        "top_users": top_users,
        "period_hours": hours
    }

def get_recent_errors(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get recent API errors.

    Args:
        limit: Maximum number of errors to return

    Returns:
        List of error log entries
    """
    query = """
        SELECT
            l.id,
            l.date_time,
            u.username,
            l.api_route,
            l.response_status_code,
            l.error_message,
            l.request_payload
        FROM user_api_call_log l
        JOIN "user" u ON l.user_id = u.id
        WHERE l.response_status_code >= 400
        ORDER BY l.date_time DESC
        LIMIT %s
    """

    errors = db_factory.execute_query(query, (limit,))

    # Parse JSON payloads
    for error in errors:
        try:
            if error['request_payload']:
                error['request_payload'] = json.loads(error['request_payload'])
        except json.JSONDecodeError:
            logger.warning(f"Could not parse request_payload for error log ID {error['id']}")
            pass # Keep as string if not valid JSON

    return errors

def cleanup_old_logs(days: int = 90) -> int:
    """
    Delete API logs older than specified days.

    Args:
        days: Number of days to keep logs

    Returns:
        Number of deleted records
    """
    query = """
        DELETE FROM user_api_call_log
        WHERE date_time < CURRENT_TIMESTAMP - INTERVAL '%s days'
    """

    deleted = db_factory.execute_write(query, (days,))
    logger.info(f"Cleaned up {deleted} old API log entries")

    return deleted

def get_recent_logs(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent API log entries.

    Args:
        limit: Maximum number of log entries to return (default: 10)

    Returns:
        List of recent log entries with user information
    """
    query = """
        SELECT
            l.id,
            l.date_time,
            u.username,
            l.api_route,
            l.response_status_code,
            l.response_time_ms,
            l.error_message,
            l.request_payload,
            l.response_payload
        FROM user_api_call_log l
        JOIN "user" u ON l.user_id = u.id
        ORDER BY l.date_time DESC
        LIMIT %s
    """

    logs = db_factory.execute_query(query, (limit,))

    # Parse JSON payloads and truncate if too large
    for log in logs:
        try:
            if log['request_payload']:
                payload = json.loads(log['request_payload'])
                # Truncate large payloads
                payload_str = json.dumps(payload)
                if len(payload_str) > 500:
                    log['request_payload'] = json.loads(payload_str[:500] + '...')
                else:
                    log['request_payload'] = payload
            else:
                log['request_payload'] = None
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Could not parse request_payload for log ID {log['id']}")
            log['request_payload'] = None

        try:
            if log['response_payload']:
                payload = json.loads(log['response_payload'])
                # Truncate large payloads
                payload_str = json.dumps(payload)
                if len(payload_str) > 500:
                    log['response_payload'] = json.loads(payload_str[:500] + '...')
                else:
                    log['response_payload'] = payload
            else:
                log['response_payload'] = None
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Could not parse response_payload for log ID {log['id']}")
            log['response_payload'] = None

        # Convert timestamp to string for API response
        log['date_time'] = str(log['date_time'])

    return logs