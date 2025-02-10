from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd
import re
from typing import List, Dict, Any, Optional
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
            # first, get the sheet names
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
