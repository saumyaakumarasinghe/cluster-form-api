from googleapiclient.discovery import build
from google.oauth2 import service_account
import re
from src.constants.link_constants import SERVICE_ACCOUNT_FILE, SCOPES, SPREADSHEET_RANGE
from src.constants.response_constants import EMPTY_LINK, INVALID_LINK, DATA_NOT_FOUND
from src.middleware.response_handler_middleware import error_response, success_response


def process_spreadsheet_link(link):
    """
    Processes a Google Sheets link, validates it, extracts the spreadsheet ID,
    and retrieves the data from the specified range in the sheet.
    """
    # validate the link
    if not link:
        return error_response(EMPTY_LINK)

    if "http" not in link:
        return error_response(INVALID_LINK)

    # extract the spreadsheet ID
    spreadsheet_id = extract_spreadsheet_id(link)
    if spreadsheet_id is None:
        return error_response(INVALID_LINK)

    # authenticate and fetch spreadsheet data
    try:
        values = fetch_spreadsheet_data(spreadsheet_id, SPREADSHEET_RANGE)
        if not values:
            return error_response(DATA_NOT_FOUND)

        # perform clustering or other processing here

        # write the processed data back to the spreadsheet
        write_to_spreadsheet(values, spreadsheet_id)

    except Exception as e:
        return error_response(f"Error fetching spreadsheet data: {str(e)}")

    return success_response(link, spreadsheet_id, values)


def extract_spreadsheet_id(link):
    """
    Extracts the spreadsheet ID from a given Google Sheets link.

    Args:
        link (str): The Google Sheets link.

    Returns:
        str: The extracted spreadsheet ID, or None if the format is invalid.
    """
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", link)
    return match.group(1) if match else None


def fetch_spreadsheet_data(spreadsheet_id, range_name):
    """
    Fetches data from a Google Sheet using the Sheets API.

    Args:
        spreadsheet_id (str): The ID of the Google Spreadsheet.
        range_name (str): The range of cells to fetch.

    Returns:
        list: The values retrieved from the specified range of the spreadsheet.
    """
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    result = (
        sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    )
    return result.get("values", [])


def write_to_spreadsheet(values, spreadsheet_id):
    """
    Writes processed data back to the specified Google Sheet.

    Args:
        values (list): The data to be written to the spreadsheet.
        spreadsheet_id (str): The ID of the Google Spreadsheet.
    """
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    (
        sheet.values()
        .update(
            spreadsheetId=spreadsheet_id,
            range="Sheet1!A1",  # you can dynamically set the range if needed
            valueInputOption="USER_ENTERED",
            body={"values": values},
        )
        .execute()
    )
