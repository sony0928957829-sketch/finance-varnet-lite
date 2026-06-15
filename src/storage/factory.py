from __future__ import annotations

from pathlib import Path

from src.utils.config import PROJECT_ROOT

from .base import ObjectStorageBackend
from .google_drive_backend import GoogleDriveObjectStorage
from .local_backend import LocalObjectStorage


def create_storage_backend(config: dict, backend_name: str | None = None) -> ObjectStorageBackend:
    remote = config.get("remote", {})
    selected = backend_name or remote.get("backend", "google_drive")
    if selected == "google_drive":
        return GoogleDriveObjectStorage.from_environment(
            remote.get("google_drive", {})
        )
    if selected == "local":
        root = Path(remote.get("local", {}).get("root", "data/cloud_mirror"))
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        return LocalObjectStorage(root)
    raise ValueError(f"Unsupported storage backend: {selected}")
