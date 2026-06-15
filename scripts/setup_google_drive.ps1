param(
    [Parameter(Mandatory = $true)]
    [string]$ClientSecrets,

    [string]$Repository = "sony0928957829-sketch/finance-varnet-lite",

    [string]$Python = ".\.venv\Scripts\python.exe",

    [switch]$SkipInitialReport
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$clientSecretsPath = [System.IO.Path]::GetFullPath($ClientSecrets)
$pythonPath = if ([System.IO.Path]::IsPathRooted($Python)) {
    $Python
} else {
    Join-Path $projectRoot $Python
}
$credentialPath = Join-Path $projectRoot ".secrets\google-drive-oauth.json"

if (-not (Test-Path -LiteralPath $clientSecretsPath)) {
    throw "OAuth client JSON was not found: $clientSecretsPath"
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python was not found: $pythonPath"
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI is required and must be authenticated before setup."
}

Push-Location $projectRoot
try {
    & $pythonPath -c "import googleapiclient, google_auth_oauthlib" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing Google Drive dependencies..."
        & $pythonPath -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install project requirements."
        }
    }

    & $pythonPath -m src.storage.auth `
        --client-secrets $clientSecretsPath `
        --output $credentialPath
    if ($LASTEXITCODE -ne 0) {
        throw "Google Drive authorization failed."
    }

    Get-Content -LiteralPath $credentialPath -Raw |
        gh secret set GOOGLE_DRIVE_OAUTH_JSON --repo $Repository
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to store GOOGLE_DRIVE_OAUTH_JSON in GitHub."
    }

    $env:GOOGLE_DRIVE_OAUTH_JSON = Get-Content -LiteralPath $credentialPath -Raw
    if (-not $SkipInitialReport) {
        Write-Host "Generating the initial real-data history..."
        & $pythonPath -m src.main --mode yfinance
        if ($LASTEXITCODE -ne 0) {
            throw "The initial yfinance pipeline failed."
        }
    }

    $output = & $pythonPath -m src.storage.cli push 2>&1
    $output | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        throw "The initial Google Drive upload failed."
    }

    $rootLine = $output |
        Where-Object { $_ -match "^Google Drive root folder id: " } |
        Select-Object -Last 1
    if (-not $rootLine) {
        throw "Google Drive root folder ID was not returned."
    }
    $rootId = ($rootLine -replace "^Google Drive root folder id: ", "").Trim()
    gh variable set GOOGLE_DRIVE_ROOT_FOLDER_ID `
        --repo $Repository `
        --body $rootId
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to store GOOGLE_DRIVE_ROOT_FOLDER_ID in GitHub."
    }

    Write-Host ""
    Write-Host "Google Drive historical storage is configured."
    Write-Host "Repository: $Repository"
    Write-Host "Drive folder ID: $rootId"
}
finally {
    Remove-Item Env:\GOOGLE_DRIVE_OAUTH_JSON -ErrorAction SilentlyContinue
    Pop-Location
}
