from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class DataLakeManifest:
    version: int
    generated_at: str
    files: tuple[ManifestEntry, ...]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "files": [asdict(item) for item in self.files],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DataLakeManifest":
        return cls(
            version=int(payload.get("version", 1)),
            generated_at=str(payload.get("generated_at", "")),
            files=tuple(
                ManifestEntry(
                    path=str(item["path"]),
                    sha256=str(item["sha256"]),
                    size=int(item["size"]),
                )
                for item in payload.get("files", [])
            ),
        )

    def by_path(self) -> dict[str, ManifestEntry]:
        return {item.path: item for item in self.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    root: Path,
    *,
    exclude_patterns: tuple[str, ...] = (),
    allowed_layers: tuple[str, ...] = (),
) -> DataLakeManifest:
    root = root.resolve()
    files = []
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            if allowed_layers and relative.split("/", 1)[0] not in allowed_layers:
                continue
            if any(fnmatch.fnmatch(relative, pattern) for pattern in exclude_patterns):
                continue
            files.append(
                ManifestEntry(
                    path=relative,
                    sha256=sha256_file(path),
                    size=path.stat().st_size,
                )
            )
    return DataLakeManifest(
        version=1,
        generated_at=datetime.now(timezone.utc).isoformat(),
        files=tuple(files),
    )


def read_manifest(path: Path) -> DataLakeManifest | None:
    if not path.exists():
        return None
    return DataLakeManifest.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def write_manifest(manifest: DataLakeManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
