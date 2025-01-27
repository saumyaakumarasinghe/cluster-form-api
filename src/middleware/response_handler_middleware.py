def error_response(message):
    """
    Creates a standardized error response.

    Args:
        message (str): The error message.

    Returns:
        tuple: A JSON response and HTTP status code.
    """
    return {"error": "true", "message": message}, 400


def success_response(link, spreadsheet_id, values):
    """
    Creates a standardized success response.

    Args:
        link (str): The original Google Sheets link.
        spreadsheet_id (str): The extracted spreadsheet ID.
        values (list): The data retrieved from the spreadsheet.

    Returns:
        tuple: A JSON response and HTTP status code.
    """
    return {
        "error": "false",
        "link": link,
        "spreadsheet_id": spreadsheet_id,
        "values": values,
    }, 200
