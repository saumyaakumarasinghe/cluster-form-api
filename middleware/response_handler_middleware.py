"""
Response Handler Middleware for standardizing API responses.
Provides consistent error and success response formats.
"""


def error_response(message):
    """
    Create a standardized error response.

    Args:
        message (str): The error message to return

    Returns:
        tuple: A JSON response and HTTP status code (400)
    """
    return {"error": "true", "message": message}, 400


def success_response(values):
    """
    Create a standardized success response.

    Args:
        values (dict): The data to return in the response

    Returns:
        tuple: A JSON response and HTTP status code (200)
    """
    return {
        "error": "false",
        "data": values,
    }, 200
