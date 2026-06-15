from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath

from .base import ObjectStorageBackend


class LocalObjectStorage(ObjectStorageBackend):
    """Filesystem mirror used for tests and offline development."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()

    def _path(self, remote_path: str) -> Path:
        relative = PurePosixPath(remote_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe remote path: {remote_path}")
        return self.root.joinpath(*relative.parts)

    def upload(self, local_path: Path, remote_path: str) -> None:
        target = self._path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, target)

    def download(self, remote_path: str, local_path: Path) -> bool:
        source = self._path(remote_path)
        if not source.exists():
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, local_path)
        return True
