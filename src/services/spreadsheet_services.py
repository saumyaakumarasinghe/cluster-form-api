from googleapiclient.discovery import build
from google.oauth2 import service_account
from openpyxl import Workbook
import re
from src.constants.link_constants import SERVICE_ACCOUNT_FILE, SCOPES, SPREADSHEET_RANGE
from src.constants.response_constants import EMPTY_LINK, INVALID_LINK, DATA_NOT_FOUND
from src.middleware.response_handler_middleware import error_response, success_response
from src.services.clustering_services import cluster_sentences


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

        # process the spreadsheet data into a list of sentences
        # assuming each row contains a sentence in the first column
        sentences = []
        for row in values:
            if row and isinstance(
                row[0], str
            ):  # check if row exists and first element is string
                sentences.append(row[0])

        # perform clustering on processed sentences
        clusters = cluster_sentences(sentences)

        # TODO store clusters in a google sheet

    except Exception as e:
        return error_response(f"Error processing spreadsheet data: {str(e)}")

    # return only link, spreadsheet_id, and clusters
    return success_response(link, spreadsheet_id, clusters)


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
