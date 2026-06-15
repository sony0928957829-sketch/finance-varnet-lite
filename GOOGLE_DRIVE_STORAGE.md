# Google Drive Historical Data Lake

VARnet-lite keeps code and lightweight reports in GitHub. Large, growing
financial history is stored as partitioned Parquet files in one app-owned
Google Drive folder.

## Data ownership

The integration requests only:

```text
https://www.googleapis.com/auth/drive.file
```

This scope lets VARnet-lite create and manage files that it created. It does
not grant broad access to unrelated personal Drive files.

Never commit OAuth client secrets, refresh tokens, or generated credential
files. The `.secrets/` directory is ignored by Git.

## Layout

```text
VARnet-lite-data/
|-- lake/
|   |-- raw/
|   |-- normalized/
|   |-- features/
|   |-- labels/
|   |-- predictions/
|   |-- alternative/
|   |-- evaluation/
|   `-- models/
`-- manifests/
    `-- data-lake-manifest.json
```

Price-like data is partitioned by dataset, market, symbol, timeframe, year,
and month. Existing partitions are merged by stable observation keys before
upload. Daily prediction snapshots include:

- `prediction_as_of`
- `input_cutoff`
- `model_version`
- `data_version`

Future label columns are excluded from prediction snapshots.

## One-time authorization

1. In Google Cloud Console, create a project and enable Google Drive API.
2. Configure an OAuth consent screen for an external or internal desktop app.
   For a personal external app, add your Google account as a test user while
   setting it up, then publish the app to production before relying on the
   daily schedule. Google testing-mode refresh tokens can expire after seven
   days.
3. Create an OAuth client with application type `Desktop app`.
4. Download the client JSON without committing it to Git.
5. Install project requirements.
6. Run:

```powershell
python -m src.storage.auth --client-secrets C:\path\to\client_secret.json
```

The browser asks the Google account owner to approve the narrow `drive.file`
scope. The helper writes `.secrets/google-drive-oauth.json`.

Store the complete contents of that file as the GitHub Actions secret:

```text
GOOGLE_DRIVE_OAUTH_JSON
```

Optionally store the created folder ID as the repository variable:

```text
GOOGLE_DRIVE_ROOT_FOLDER_ID
```

If the variable is omitted, the integration finds or creates its app-owned
`VARnet-lite-data` folder.

On Windows, the repository includes a setup helper that performs the
authorization, GitHub Secret creation, initial upload, and repository-variable
creation in one run:

```powershell
.\scripts\setup_google_drive.ps1 `
  -ClientSecrets C:\path\to\client_secret.json
```

The only interactive part is Google's own account-consent page.
By default, the helper then runs the yfinance pipeline, uploads the first real
history, stores the OAuth JSON as a GitHub Secret, and records the Drive folder
ID as a repository variable.

## Daily behavior

The workflow performs these operations in order:

1. Pull and verify the remote manifest.
2. Download only missing or changed partitions.
3. Run the market pipeline.
4. Merge the current observations into local partitions.
5. Upload only files whose SHA-256 changed.
6. Upload the new manifest last.

Remote deletion is deliberately disabled. A failed run therefore cannot erase
historical data.

When the secret is not configured, both Drive steps exit successfully and the
mock and yfinance pipelines continue to work locally.

Mock outputs are never included in the remote manifest. Only configured real
data modes, currently `yfinance`, are archived to the production data lake.
