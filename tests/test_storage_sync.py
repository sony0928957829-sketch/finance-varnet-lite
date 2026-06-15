from pathlib import Path
import unittest

from src.storage.local_backend import LocalObjectStorage
from src.storage.sync import pull_data_lake, push_data_lake
from src.storage.manifest import DataLakeManifest, ManifestEntry, write_manifest


def test_local_backend_round_trip_is_incremental(tmp_path: Path):
    lake = tmp_path / "lake"
    manifest = tmp_path / "manifest.json"
    remote = tmp_path / "remote"
    file_path = lake / "raw" / "prices" / "part.parquet"
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"market history")
    backend = LocalObjectStorage(remote)

    first = push_data_lake(lake, manifest, "manifests/lake.json", backend)
    second = push_data_lake(lake, manifest, "manifests/lake.json", backend)

    assert first.uploaded == 1
    assert second.uploaded == 0
    assert second.unchanged == 1

    file_path.unlink()
    pulled = pull_data_lake(lake, manifest, "manifests/lake.json", backend)
    assert pulled.downloaded == 1
    assert file_path.read_bytes() == b"market history"


def test_pull_without_remote_manifest_is_a_clean_first_run(tmp_path: Path):
    result = pull_data_lake(
        tmp_path / "lake",
        tmp_path / "manifest.json",
        "manifests/lake.json",
        LocalObjectStorage(tmp_path / "remote"),
    )
    assert result.remote_missing is True


def test_pull_rejects_corrupted_remote_object(tmp_path: Path):
    lake = tmp_path / "lake"
    manifest = tmp_path / "manifest.json"
    remote = tmp_path / "remote"
    file_path = lake / "raw" / "part.parquet"
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"valid")
    backend = LocalObjectStorage(remote)
    push_data_lake(lake, manifest, "manifests/lake.json", backend)

    (remote / "lake" / "raw" / "part.parquet").write_bytes(b"corrupt")
    file_path.unlink()

    with unittest.TestCase().assertRaises(RuntimeError):
        pull_data_lake(lake, manifest, "manifests/lake.json", backend)
    assert not file_path.exists()


def test_pull_rejects_manifest_path_traversal(tmp_path: Path):
    remote = tmp_path / "remote"
    remote_manifest = remote / "manifests" / "lake.json"
    write_manifest(
        DataLakeManifest(
            version=1,
            generated_at="2026-06-15T00:00:00Z",
            files=(ManifestEntry(path="../outside.txt", sha256="x", size=1),),
        ),
        remote_manifest,
    )

    with unittest.TestCase().assertRaises(RuntimeError):
        pull_data_lake(
            tmp_path / "lake",
            tmp_path / "manifest.json",
            "manifests/lake.json",
            LocalObjectStorage(remote),
            allowed_layers=("raw",),
        )
