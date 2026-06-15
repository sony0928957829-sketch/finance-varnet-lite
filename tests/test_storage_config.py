from pathlib import Path

from src.utils.config import load_config


def test_storage_uses_narrow_google_drive_scope():
    config = load_config("storage.yaml")
    drive = config["remote"]["google_drive"]

    assert config["remote"]["backend"] == "google_drive"
    assert drive["oauth_scope"].endswith("/auth/drive.file")
    assert drive["credentials_json_env"] == "GOOGLE_DRIVE_OAUTH_JSON"
    assert config["sync"]["delete_remote_files"] is False
    assert config["data_lake"]["archive_modes"] == ["yfinance"]
    assert "**/*_mock/**" in config["sync"]["exclude_globs"]

    gitignore = (
        Path(__file__).resolve().parents[1] / ".gitignore"
    ).read_text(encoding="utf-8")
    assert ".secrets/" in gitignore


def test_workflow_syncs_before_and_after_pipeline():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "daily-market-report.yml"
    ).read_text(encoding="utf-8")

    pull = "python -m src.storage.cli pull --if-configured"
    run = "python -m src.main --mode yfinance"
    push = "python -m src.storage.cli push --if-configured"
    assert pull in workflow
    assert push in workflow
    assert workflow.index(pull) < workflow.index(run) < workflow.index(push)
    assert "secrets.GOOGLE_DRIVE_OAUTH_JSON" in workflow
    assert "timeout-minutes: 60" in workflow
