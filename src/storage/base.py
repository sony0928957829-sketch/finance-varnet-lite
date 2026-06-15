from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ObjectStorageBackend(ABC):
    """Minimal object interface used by data-lake synchronization."""

    @abstractmethod
    def upload(self, local_path: Path, remote_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def download(self, remote_path: str, local_path: Path) -> bool:
        """Download an object and return False when it does not exist."""
        raise NotImplementedError
