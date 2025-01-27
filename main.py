from googleapiclient.discovery import build
from google.oauth2 import service_account

SERVICE_ACCOUNT_FILE = "keys.json"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]  # read only readonly

creds = None
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

# The ID spreadsheet
SAMPLE_SPREADSHEET_ID = "1LbxrSc4axJx-gUjjzT-g8e_W9Y-kk7MtdlDgsExZOYk"

service = build("sheets", "v4", credentials=creds)

# Call the Sheets API
sheet = service.spreadsheets()
result = (
    sheet.values()
    .get(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="Student Responses!A1:J63")
    .execute()
)
values = result.get("values", [])

# aoa = [["saumya", 1000], ["saumya2", 2000], ["saumya3", 3000]]

# Write in the sheet
request = (
    sheet.values()
    .update(
        spreadsheetId=SAMPLE_SPREADSHEET_ID,
        range="Sheet1!A1",
        valueInputOption="USER_ENTERED",
        body={"values": values},
    )
    .execute()
)

print(request)
