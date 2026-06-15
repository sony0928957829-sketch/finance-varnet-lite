from __future__ import annotations

import argparse
import json
from pathlib import Path


DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"


def authorize(client_secrets: Path, output: Path) -> Path:
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Google OAuth dependencies are missing. Install requirements.txt."
        ) from exc

    flow = InstalledAppFlow.from_client_secrets_file(
        str(client_secrets),
        scopes=[DRIVE_FILE_SCOPE],
    )
    credentials = flow.run_local_server(
        host="localhost",
        port=0,
        access_type="offline",
        prompt="consent",
        open_browser=True,
    )
    payload = {
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "scopes": list(credentials.scopes or [DRIVE_FILE_SCOPE]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authorize VARnet-lite to its app-owned Google Drive files."
    )
    parser.add_argument(
        "--client-secrets",
        required=True,
        type=Path,
        help="OAuth desktop client JSON downloaded from Google Cloud Console.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".secrets/google-drive-oauth.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = authorize(args.client_secrets, args.output)
    print(f"Authorization saved securely to: {path}")
    print("Store the complete file contents in GitHub secret GOOGLE_DRIVE_OAUTH_JSON.")
