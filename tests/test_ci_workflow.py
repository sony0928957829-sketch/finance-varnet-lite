from pathlib import Path


def test_pull_request_ci_is_required_and_offline_safe():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "ci.yml"
    ).read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "python -m pytest tests -q" in workflow
    assert "python -m src.main --mode mock" in workflow
    assert "range_label_columns" in workflow
    assert "contents: read" in workflow
    assert "--mode yfinance" not in workflow
