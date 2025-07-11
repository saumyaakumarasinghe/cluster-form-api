from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd
import re
from typing import Tuple, List, Any, Optional
from googleapiclient.errors import HttpError

from constants.link_constants import SERVICE_ACCOUNT_FILE, SCOPES


class SpreadsheetService:
    """
    Service for interacting with Google Sheets using a service account.
    Handles authentication, data fetching, and writing cluster results back to the sheet.
    """

    def __init__(self):
        # Authenticate using service account credentials
        self.creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        self.service = build("sheets", "v4", credentials=self.creds)
        self.forms_service = build("forms", "v1", credentials=self.creds)
        self.service_account_email = self.creds.service_account_email

    @staticmethod
    def extract_spreadsheet_id(link: str) -> Optional[str]:
        """
        Extract the spreadsheet ID from a Google Sheets link.
        Args:
            link (str): Google Sheets URL.
        Returns:
            str or None: Spreadsheet ID if found, else None.
        """
        match = re.search(r"/d/([a-zA-Z0-9-_]+)", link)
        return match.group(1) if match else None

    def fetch_spreadsheet_data(
        self, spreadsheet_id: str, range_name: str
    ) -> List[List[str]]:
        """
        Fetch data from a Google Sheet for a given range.
        Args:
            spreadsheet_id (str): The ID of the spreadsheet.
            range_name (str): The range to fetch (e.g., 'A1:Z').
        Returns:
            list: List of rows (each row is a list of cell values).
        Raises:
            PermissionError: If the service account does not have access.
            Exception: For invalid range or other errors.
        """
        try:
            # Check if the service account has permission to access the sheet
            has_permission, error_message = self._check_permissions(spreadsheet_id)
            if not has_permission:
                raise PermissionError(
                    f"Permission denied. {error_message}. Please share the spreadsheet with {self.service_account_email}"
                )
            # Get sheet metadata and determine the correct range
            sheet_metadata = (
                self.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            )
            sheets = sheet_metadata.get("sheets", [])
            if len(sheets) == 1:
                # If only one sheet, use its name in the range
                sheet_name = sheets[0]["properties"]["title"]
                full_range = f"'{sheet_name}'!{range_name}"
            else:
                # If multiple sheets, use the provided range as is
                full_range = range_name
            # Fetch the data from the sheet
            result = (
                self.service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=full_range)
                .execute()
            )
            return result.get("values", [])
        except HttpError as e:
            if e.resp.status == 403:
                raise PermissionError(
                    f"Access denied. Please share the spreadsheet with {self.service_account_email}"
                )
            elif "Invalid range" in str(e):
                try:
                    result = (
                        self.service.spreadsheets()
                        .values()
                        .get(spreadsheetId=spreadsheet_id, range=range_name)
                        .execute()
                    )
                    return result.get("values", [])
                except Exception as inner_e:
                    raise Exception(
                        f"Could not find valid data in range {range_name}. Please check your spreadsheet structure."
                    )
            raise e

    def convert_to_dataframe(
        self, data: List[List[Any]], column_name: str
    ) -> pd.DataFrame:
        """
        Convert sheet data to a pandas DataFrame with a single column.
        Args:
            data (list): List of rows from the sheet.
            column_name (str): Name to use for the DataFrame column.
        Returns:
            pd.DataFrame: DataFrame with the specified column.
        Raises:
            ValueError: If no data is found.
        """
        if not data:
            raise ValueError("No data found in spreadsheet")
        return pd.DataFrame(data, columns=[column_name])

    def _check_permissions(self, spreadsheet_id: str) -> Tuple[bool, str]:
        """
        Check if the service account has required permissions to access the sheet.
        Args:
            spreadsheet_id (str): The ID of the spreadsheet.
        Returns:
            tuple: (True, '') if access is granted, (False, error message) otherwise.
        """
        try:
            self.service.spreadsheets().get(
                spreadsheetId=spreadsheet_id, fields="spreadsheetId"
            ).execute()
            return True, ""
        except HttpError as e:
            if e.resp.status == 403:
                return (
                    False,
                    f"Spreadsheet access denied. Please share it with {self.service_account_email}",
                )
            return False, str(e)

    def write_dataframe_to_spreadsheet(self, spreadsheet_id: str, df: pd.DataFrame):
        """
        Write clustered results back to the Google Sheet, creating a new sheet for each cluster.
        If a sheet for a cluster already exists, overwrite its contents.
        Args:
            spreadsheet_id (str): The ID of the spreadsheet.
            df (pd.DataFrame): DataFrame containing the data and cluster labels.
        Returns:
            str: Link to the updated spreadsheet.
        Raises:
            Exception: If writing fails.
        """
        try:
            if "cluster" not in df.columns:
                raise ValueError("Dataframe must contain a 'cluster' column")
            # Group data by cluster and write each group to a new sheet
            cluster_groups = df.groupby("cluster")
            for cluster, data in cluster_groups:
                sheet_name = f"Cluster {cluster}"
                # Try to add a new sheet; if it exists, clear it
                try:
                    self.service.spreadsheets().batchUpdate(
                        spreadsheetId=spreadsheet_id,
                        body={
                            "requests": [
                                {"addSheet": {"properties": {"title": sheet_name}}}
                            ]
                        },
                    ).execute()
                except HttpError as e:
                    if "already exists" not in str(e):
                        raise e
                    # If the sheet exists, clear its contents
                    self.service.spreadsheets().values().clear(
                        spreadsheetId=spreadsheet_id, range=f"{sheet_name}!A:Z"
                    ).execute()
                # Write the data to the new or cleared sheet
                values = [data.columns.tolist()] + data.values.tolist()
                self.service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=f"{sheet_name}!A1",
                    valueInputOption="RAW",
                    body={"values": values},
                ).execute()
            return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
        except Exception as e:
            raise Exception(f"Error creating cluster sheets: {str(e)}")
