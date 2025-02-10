def error_response(message):
    """
    Creates a standardized error response.

    Args:
        message (str): The error message.

    Returns:
        tuple: A JSON response and HTTP status code.
    """
    return {"error": "true", "message": message}, 400


def success_response(values):

    return {
        "error": "false",
        "data": values,
    }, 200
