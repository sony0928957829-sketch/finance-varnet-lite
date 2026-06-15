param(
    [string]$ClientSecrets = "",

    [string]$Repository = "sony0928957829-sketch/finance-varnet-lite",

    [string]$Python = ".\.venv\Scripts\python.exe",

    [switch]$SkipInitialReport
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = if ([System.IO.Path]::IsPathRooted($Python)) {
    $Python
} else {
    Join-Path $projectRoot $Python
}
$credentialPath = Join-Path $projectRoot ".secrets\google-drive-oauth.json"
$ghCommand = Get-Command gh -ErrorAction SilentlyContinue
$ghPath = if ($ghCommand) {
    $ghCommand.Source
} else {
    $toolsRoot = Join-Path (Split-Path -Parent $projectRoot) ".tools"
    Get-ChildItem -LiteralPath $toolsRoot `
        -Filter "gh.exe" `
        -File `
        -Recurse `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}

function Resolve-ClientSecretsPath {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        $resolved = [System.IO.Path]::GetFullPath($RequestedPath)
        if (Test-Path -LiteralPath $resolved) {
            return $resolved
        }
        throw "OAuth client JSON was not found: $resolved"
    }

    $downloads = Join-Path ([Environment]::GetFolderPath("UserProfile")) "Downloads"
    $downloaded = Get-ChildItem -LiteralPath $downloads `
        -Filter "client_secret*.json" `
        -File `
        -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($downloaded) {
        Write-Host "Using Google OAuth file: $($downloaded.FullName)"
        return $downloaded.FullName
    }

    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = "Select the Google Desktop OAuth JSON file"
    $dialog.Filter = "Google OAuth JSON (*.json)|*.json"
    $dialog.InitialDirectory = $downloads
    if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
        throw "Google OAuth setup was cancelled."
    }
    return $dialog.FileName
}

function Assert-DesktopClientSecrets {
    param([string]$Path)

    try {
        $json = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    }
    catch {
        throw "The selected file is not valid JSON: $Path"
    }
    if (-not $json.installed.client_id -or -not $json.installed.client_secret) {
        throw "Select a Google OAuth client created as application type 'Desktop app'."
    }
}

$clientSecretsPath = Resolve-ClientSecretsPath -RequestedPath $ClientSecrets
Assert-DesktopClientSecrets -Path $clientSecretsPath

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python was not found: $pythonPath"
}
if (-not $ghPath -or -not (Test-Path -LiteralPath $ghPath)) {
    throw "GitHub CLI is required and must be authenticated before setup."
}

Push-Location $projectRoot
try {
    & $ghPath auth status --hostname github.com *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "GitHub authorization is required. Complete the browser sign-in..."
        & $ghPath auth login `
            --hostname github.com `
            --git-protocol https `
            --web
        if ($LASTEXITCODE -ne 0) {
            throw "GitHub authorization failed."
        }
    }

    $branch = (& git branch --show-current).Trim()
    if (-not $branch) {
        throw "The current Git branch could not be determined."
    }
    Write-Host "Publishing the current setup branch..."
    & git push --set-upstream origin $branch
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to publish the current Git branch."
    }

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
        & $ghPath secret set GOOGLE_DRIVE_OAUTH_JSON --repo $Repository
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
    & $ghPath variable set GOOGLE_DRIVE_ROOT_FOLDER_ID `
        --repo $Repository `
        --body $rootId
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to store GOOGLE_DRIVE_ROOT_FOLDER_ID in GitHub."
    }

    Write-Host ""
    Write-Host "Google Drive historical storage is configured."
    Write-Host "Repository: $Repository"
    Write-Host "Drive folder ID: $rootId"

    if ($branch -ne "main") {
        $pullRequestUrl = & $ghPath pr list `
            --repo $Repository `
            --head $branch `
            --state open `
            --json url `
            --jq ".[0].url"
        if (-not $pullRequestUrl) {
            $pullRequestUrl = & $ghPath pr create `
                --repo $Repository `
                --base main `
                --head $branch `
                --title "feat: add Google Drive historical data lake" `
                --body "Adds partitioned Google Drive historical storage, incremental synchronization, integrity checks, and GitHub Actions integration."
            if ($LASTEXITCODE -ne 0) {
                throw "Google Drive setup finished, but the pull request could not be created."
            }
        }
        Write-Host "Pull request: $pullRequestUrl"
    }
}
finally {
    Remove-Item Env:\GOOGLE_DRIVE_OAUTH_JSON -ErrorAction SilentlyContinue
    Pop-Location
}
