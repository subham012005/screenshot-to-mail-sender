"""
Run this script ONCE locally to generate token.json for Gmail API.
Then upload token.json as a Secret File on Render at path: /etc/secrets/token.json

Steps:
  1. Go to https://console.cloud.google.com/
  2. Create a project → Enable Gmail API
  3. Create OAuth 2.0 credentials (Desktop App) → Download credentials.json
  4. Place credentials.json next to this script
  5. Run: python generate_token.py
  6. A browser window opens → sign in and grant permission
  7. token.json is created → upload it to Render as a Secret File
"""

from google_auth_oauthlib.flow import InstalledAppFlow
import json, os

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def main():
    creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.json")
    if not os.path.exists(creds_path):
        print("ERROR: credentials.json not found.")
        print("Download it from Google Cloud Console → APIs & Services → Credentials")
        return

    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
    creds = flow.run_local_server(port=0)

    token_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token.json")
    with open(token_path, "w") as f:
        f.write(creds.to_json())

    print(f"\nSUCCESS: token.json created at: {token_path}")
    print("Now upload it to Render:")
    print("  Render Dashboard → Your Service → Environment → Secret Files → Add file")
    print("  Filename: token.json  |  Path: /etc/secrets/token.json")
    print("  Paste the contents of token.json into the file body.")

if __name__ == "__main__":
    main()
