from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd
import re
from typing import Tuple, List, Any, Optional
from googleapiclient.errors import HttpError

from constants.link_constants import SERVICE_ACCOUNT_FILE, SCOPES


class SpreadsheetService:
    def __init__(self):
        self.creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        self.service = build("sheets", "v4", credentials=self.creds)
        self.drive_service = build("drive", "v3", credentials=self.creds)
        self.service_account_email = self.creds.service_account_email

    @staticmethod
    def extract_spreadsheet_id(link: str) -> Optional[str]:
        """Extract spreadsheet ID from Google Sheets link"""
        match = re.search(r"/d/([a-zA-Z0-9-_]+)", link)
        return match.group(1) if match else None

    def fetch_spreadsheet_data(
        self, spreadsheet_id: str, range_name: str
    ) -> List[List[str]]:
        try:
            has_permission, error_message = self._check_permissions(spreadsheet_id)
            if not has_permission:
                raise PermissionError(
                    f"Permission denied. {error_message}. Please share the spreadsheet with {self.service_account_email}"
                )

            sheet_metadata = (
                self.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            )
            sheets = sheet_metadata.get("sheets", [])

            if len(sheets) == 1:
                sheet_name = sheets[0]["properties"]["title"]
                full_range = f"'{sheet_name}'!{range_name}"
            else:
                full_range = range_name

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
        """Convert sheet data to pandas DataFrame and prepare a single column for clustering"""
        if not data:
            raise ValueError("No data found in spreadsheet")
        return pd.DataFrame(data, columns=[column_name])

    def _check_permissions(self, spreadsheet_id: str) -> Tuple[bool, str]:
        """Check if the service account has required permissions"""
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
        """Creates separate sheets for each cluster in the given spreadsheet"""
        try:
            if "cluster" not in df.columns:
                raise ValueError("Dataframe must contain a 'cluster' column")

            cluster_groups = df.groupby("cluster")
            for cluster, data in cluster_groups:
                sheet_name = f"Cluster {cluster}"
                request = {"addSheet": {"properties": {"title": sheet_name}}}
                try:
                    self.service.spreadsheets().batchUpdate(
                        spreadsheetId=spreadsheet_id, body={"requests": [request]}
                    ).execute()
                except HttpError as e:
                    if "already exists" not in str(e):
                        raise e

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
