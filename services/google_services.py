from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd
import re
from typing import Tuple
from googleapiclient.errors import HttpError

from typing import List, Any, Optional
from googleapiclient.errors import HttpError
from constants.link_constants import SERVICE_ACCOUNT_FILE, SCOPES


class SpreadsheetService:
    def __init__(self):
        self.creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        self.service = build("sheets", "v4", credentials=self.creds)
        self.forms_service = build("forms", "v1", credentials=self.creds)

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
                except HttpError as e:
                    if e.resp.status == 400:
                        raise Exception(
                            f"Invalid range: {range_name}. Check spreadsheet structure."
                        )
                    raise e

    def get_linked_google_form(self, spreadsheet_id: str) -> Optional[str]:
        """Check if the spreadsheet is linked to a Google Form and return the Form ID."""
        try:
            sheet_metadata = (
                self.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            )
            sheets = sheet_metadata.get("sheets", [])
            for sheet in sheets:
                form_id = sheet.get("properties", {}).get("sourceForm", None)
                if form_id:
                    return form_id
            return None
        except Exception as e:
            print(f"Error checking for linked form: {e}")
            return None

    def get_form_questions(self, form_id: str) -> dict:
        """Fetch form questions and identify open-ended ones."""
        try:
            form_info = self.forms_service.forms().get(formId=form_id).execute()
            questions = {}
            for item in form_info.get("items", []):
                question_id = (
                    item.get("questionItem", {}).get("question", {}).get("questionId")
                )
                question_type = (
                    item.get("questionItem", {}).get("question", {}).get("type")
                )
                if question_id and question_type:
                    questions[question_id] = question_type
            return {qid: qtype for qid, qtype in questions.items() if qtype == "TEXT"}
        except Exception as e:
            print(f"Error retrieving form questions: {e}")
            return {}

    def detect_open_ended_columns(self, sheet_data: List[List[str]]) -> List[str]:
        """Detect possible open-ended text columns based on response length and variety."""
        if not sheet_data or len(sheet_data) < 2:
            return []
        headers = sheet_data[0]
        rows = sheet_data[1:]
        open_ended_columns = []
        for col_idx, col_name in enumerate(headers):
            col_data = [row[col_idx] for row in rows if len(row) > col_idx]
            if not col_data:
                continue
            avg_word_count = np.mean([len(str(cell).split()) for cell in col_data])
            unique_ratio = len(set(col_data)) / len(col_data)
            if avg_word_count > 5 and unique_ratio > 0.6:
                open_ended_columns.append(col_name)
        return open_ended_columns

    def identify_open_ended_column(
        self, spreadsheet_link: str, form_link: Optional[str], accuracy_required: bool
    ) -> List[str]:
        """Determine which column in the spreadsheet contains open-ended responses."""
        spreadsheet_id = self.extract_spreadsheet_id(spreadsheet_link)
        if not spreadsheet_id:
            raise ValueError("Invalid spreadsheet link")
        form_id = self.get_linked_google_form(spreadsheet_id)

        if form_id:
            form_questions = self.get_form_questions(form_id)
            return list(form_questions.keys())
        elif accuracy_required and form_link:
            form_id = self.extract_spreadsheet_id(form_link)
            if form_id:
                form_questions = self.get_form_questions(form_id)
                return list(form_questions.keys())
        else:
            sheet_data = self.fetch_spreadsheet_data(spreadsheet_id, "A1:Z")
            return self.detect_open_ended_columns(sheet_data)

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
