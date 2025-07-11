"""
Health Controller for checking API server status.
Provides basic health check functionality.
"""


def health_check():
    """
    Check if the server is running and healthy.

    Returns:
        dict: Status information about the server
    """
    return {"status": "OK", "message": "Server is healthy"}
