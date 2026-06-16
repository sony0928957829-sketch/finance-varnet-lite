from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import tempfile

from .base import ObjectStorageBackend
from .manifest import (
    DataLakeManifest,
    build_manifest,
    read_manifest,
    sha256_file,
    write_manifest,
)


@dataclass(frozen=True)
class SyncResult:
    uploaded: int = 0
    downloaded: int = 0
    unchanged: int = 0
    remote_missing: bool = False


def push_data_lake(
    local_root: Path,
    local_manifest_path: Path,
    remote_manifest_path: str,
    backend: ObjectStorageBackend,
    *,
    exclude_patterns: tuple[str, ...] = (),
    allowed_layers: tuple[str, ...] = (),
) -> SyncResult:
    manifest = build_manifest(
        local_root,
        exclude_patterns=exclude_patterns,
        allowed_layers=allowed_layers,
    )
    remote_manifest = _download_remote_manifest(backend, remote_manifest_path)
    remote_by_path = remote_manifest.by_path() if remote_manifest else {}

    uploaded = 0
    unchanged = 0
    for entry in manifest.files:
        if (
            entry.path in remote_by_path
            and remote_by_path[entry.path].sha256 == entry.sha256
        ):
            unchanged += 1
            continue
        backend.upload(
            _local_object_path(local_root, entry.path, allowed_layers),
            f"lake/{entry.path}",
        )
        uploaded += 1

    write_manifest(manifest, local_manifest_path)
    backend.upload(local_manifest_path, remote_manifest_path)
    return SyncResult(uploaded=uploaded, unchanged=unchanged)


def pull_data_lake(
    local_root: Path,
    local_manifest_path: Path,
    remote_manifest_path: str,
    backend: ObjectStorageBackend,
    *,
    verify_sha256: bool = True,
    allowed_layers: tuple[str, ...] = (),
) -> SyncResult:
    remote_manifest = _download_remote_manifest(backend, remote_manifest_path)
    if remote_manifest is None:
        return SyncResult(remote_missing=True)

    local_manifest = build_manifest(local_root)
    local_by_path = local_manifest.by_path()
    downloaded = 0
    unchanged = 0

    for entry in remote_manifest.files:
        local_entry = local_by_path.get(entry.path)
        if local_entry and local_entry.sha256 == entry.sha256:
            unchanged += 1
            continue
        target = _local_object_path(local_root, entry.path, allowed_layers)
        if not backend.download(f"lake/{entry.path}", target):
            raise RuntimeError(f"Remote manifest references a missing file: {entry.path}")
        if verify_sha256 and sha256_file(target) != entry.sha256:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 verification failed: {entry.path}")
        downloaded += 1

    write_manifest(remote_manifest, local_manifest_path)
    return SyncResult(downloaded=downloaded, unchanged=unchanged)


def _download_remote_manifest(
    backend: ObjectStorageBackend,
    remote_manifest_path: str,
) -> DataLakeManifest | None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        if not backend.download(remote_manifest_path, path):
            return None
        return read_manifest(path)


def _local_object_path(
    root: Path,
    relative_path: str,
    allowed_layers: tuple[str, ...] = (),
) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError(f"Unsafe manifest path: {relative_path}")
    if allowed_layers and relative.parts[0] not in allowed_layers:
        raise RuntimeError(f"Manifest path uses a disabled layer: {relative_path}")
    root = root.resolve()
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root):
        raise RuntimeError(f"Manifest path escapes the data lake: {relative_path}")
    return target
