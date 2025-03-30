from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd
import re
from typing import Tuple
from googleapiclient.errors import HttpError

from typing import List, Any, Optional
from constants.link_constants import SERVICE_ACCOUNT_FILE, SCOPES


class SpreadsheetService:
    def __init__(self):
        self.creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        self.service = build("sheets", "v4", credentials=self.creds)

    @staticmethod
    def extract_spreadsheet_id(link: str) -> Optional[str]:
        """Extract spreadsheet ID from Google Sheets link"""
        match = re.search(r"/d/([a-zA-Z0-9-_]+)", link)
        return match.group(1) if match else None

    def fetch_spreadsheet_data(
        self, spreadsheet_id: str, range_name: str
    ) -> List[List[str]]:
        try:
            # First check permissions
            has_permission, error_message = self._check_permissions(spreadsheet_id)
            if not has_permission:
                raise PermissionError(f"Permission denied. {error_message}. ")

            # get the sheet names
            sheet_metadata = (
                self.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            )
            sheets = sheet_metadata.get("sheets", [])

            # if there's only one sheet, use it directly
            if len(sheets) == 1:
                sheet_name = sheets[0]["properties"]["title"]
                full_range = f"'{sheet_name}'!{range_name}"
            else:
                # if multiple sheets, try the range as is first
                full_range = range_name

            result = (
                self.service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=full_range)
                .execute()
            )

            values = result.get("values", [])
            return values

        except Exception as e:
            if "Invalid range" in str(e):
                # try without sheet name if that fails
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
        """Convert sheet data to pandas DataFrame and prepare a single column for clustering"""
        if not data:
            raise ValueError("No data found in spreadsheet")

        # handle column by directly using the provided column_name
        return pd.DataFrame(data, columns=[column_name])

    def _check_permissions(self, spreadsheet_id: str) -> Tuple[bool, str]:
        """
        Check if the service account has required permissions
        """
        try:
            # Try to get minimal metadata to check permissions
            self.service.spreadsheets().get(
                spreadsheetId=spreadsheet_id, fields="spreadsheetId"
            ).execute()
            return True, ""
        except HttpError as e:
            if e.resp.status == 403:
                return (
                    False,
                    "You need to share the spreadsheet with the service account email or make it public",
                )
            return False, str(e)

    def create_spreadsheet(self, title):
        """
        Creates a new Google Spreadsheet

        Args:
            title (str): Title for the new spreadsheet

        Returns:
            str: ID of the newly created spreadsheet
        """
        try:
            # Create a new spreadsheet
            spreadsheet_body = {"properties": {"title": title}}

            spreadsheet = (
                self.service.spreadsheets().create(body=spreadsheet_body).execute()
            )

            spreadsheet_id = spreadsheet.get("spreadsheetId")

            # Set permissions to anyone with the link can view
            self.drive_service.permissions().create(
                fileId=spreadsheet_id, body={"type": "anyone", "role": "reader"}
            ).execute()

            return spreadsheet_id

        except Exception as e:
            print(f"Error creating spreadsheet: {str(e)}")
            raise

    def write_dataframe_to_spreadsheet(self, spreadsheet_id, df):
        """
        Writes a pandas DataFrame to a Google Spreadsheet

        Args:
            spreadsheet_id (str): ID of the spreadsheet
            df (pandas.DataFrame): DataFrame to write

        Returns:
            str: Link to the spreadsheet
        """
        try:
            # Convert dataframe to list of lists for sheets API
            values = [df.columns.tolist()]  # Header row
            values.extend(df.values.tolist())  # Data rows

            # Determine the range based on dataframe dimensions
            range_name = f"Sheet1!A1:{chr(64 + len(df.columns))}{len(df) + 1}"

            # Clear any existing data
            self.service.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id, range=range_name
            ).execute()

            # Write the data
            self.service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range="Sheet1!A1",
                valueInputOption="RAW",
                body={"values": values},
            ).execute()

            # Create a summary sheet with clustering information
            self._create_summary_sheet(spreadsheet_id, df)

            # Return the link to the spreadsheet
            return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"

        except Exception as e:
            print(f"Error writing to spreadsheet: {str(e)}")
            raise

    def _create_summary_sheet(self, spreadsheet_id, df):
        """
        Creates a summary sheet with cluster information

        Args:
            spreadsheet_id (str): ID of the spreadsheet
            df (pandas.DataFrame): DataFrame with cluster results
        """
        try:
            # Create a new sheet for summary
            request = {"addSheet": {"properties": {"title": "Cluster Summary"}}}

            self.service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id, body={"requests": [request]}
            ).execute()

            # Calculate cluster stats
            if "cluster" in df.columns:
                cluster_counts = df["cluster"].value_counts().reset_index()
                cluster_counts.columns = ["Cluster", "Count"]

                # Convert to values list
                summary_values = [["Cluster", "Count"]]
                summary_values.extend(cluster_counts.values.tolist())

                # Write summary data
                self.service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range="Cluster Summary!A1",
                    valueInputOption="RAW",
                    body={"values": summary_values},
                ).execute()

        except Exception as e:
            print(f"Error creating summary sheet: {str(e)}")
            # Don't raise - this is a non-critical feature
