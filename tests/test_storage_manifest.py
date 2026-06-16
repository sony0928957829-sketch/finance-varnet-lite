from pathlib import Path

from src.storage.manifest import build_manifest, read_manifest, write_manifest


def test_manifest_is_stable_and_detects_changes(tmp_path: Path):
    root = tmp_path / "lake"
    root.mkdir()
    path = root / "normalized" / "part.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"first")

    first = build_manifest(root)
    second = build_manifest(root)

    assert first.files == second.files
    assert first.files[0].path == "normalized/part.parquet"

    path.write_bytes(b"second")
    changed = build_manifest(root)
    assert changed.files[0].sha256 != first.files[0].sha256

    manifest_path = tmp_path / "manifest.json"
    write_manifest(changed, manifest_path)
    assert read_manifest(manifest_path).files == changed.files


def test_manifest_can_exclude_mock_outputs(tmp_path: Path):
    real = tmp_path / "features" / "features_yfinance" / "part.parquet"
    mock = tmp_path / "features" / "features_mock" / "part.parquet"
    real.parent.mkdir(parents=True)
    mock.parent.mkdir(parents=True)
    real.write_bytes(b"real")
    mock.write_bytes(b"mock")

    manifest = build_manifest(tmp_path, exclude_patterns=("**/*_mock/**",))
    assert [entry.path for entry in manifest.files] == [
        "features/features_yfinance/part.parquet"
    ]


def test_manifest_only_includes_enabled_layers(tmp_path: Path):
    allowed = tmp_path / "raw" / "part.parquet"
    unexpected = tmp_path / "private" / "secret.txt"
    allowed.parent.mkdir()
    unexpected.parent.mkdir()
    allowed.write_bytes(b"data")
    unexpected.write_bytes(b"do not upload")

    manifest = build_manifest(tmp_path, allowed_layers=("raw",))
    assert [entry.path for entry in manifest.files] == ["raw/part.parquet"]
