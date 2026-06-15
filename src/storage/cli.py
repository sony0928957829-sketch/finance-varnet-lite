from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.storage.factory import create_storage_backend
from src.storage.sync import pull_data_lake, push_data_lake
from src.utils.config import PROJECT_ROOT, load_config


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _is_configured(config: dict, backend_name: str | None) -> bool:
    selected = backend_name or config.get("remote", {}).get(
        "backend",
        "google_drive",
    )
    if selected != "google_drive":
        return True
    drive = config.get("remote", {}).get("google_drive", {})
    return bool(os.environ.get(drive.get("credentials_json_env", ""), "").strip())


def run_sync(
    action: str,
    *,
    backend_name: str | None = None,
    if_configured: bool = False,
) -> int:
    config = load_config("storage.yaml")
    if if_configured and not _is_configured(config, backend_name):
        print("Google Drive is not configured; remote sync skipped.")
        return 0

    backend = create_storage_backend(config, backend_name)
    lake = config["data_lake"]
    local_root = _resolve(lake["local_root"])
    local_manifest = _resolve(lake["manifest_path"])
    remote_manifest = config["remote"]["manifest_path"]
    allowed_layers = tuple(config.get("sync", {}).get("enabled_layers", []))

    if action == "pull":
        result = pull_data_lake(
            local_root,
            local_manifest,
            remote_manifest,
            backend,
            verify_sha256=bool(
                config.get("sync", {}).get("verify_sha256", True)
            ),
            allowed_layers=allowed_layers,
        )
    elif action == "push":
        result = push_data_lake(
            local_root,
            local_manifest,
            remote_manifest,
            backend,
            exclude_patterns=tuple(
                config.get("sync", {}).get("exclude_globs", [])
            ),
            allowed_layers=allowed_layers,
        )
    else:
        raise ValueError(f"Unsupported action: {action}")

    root_id = getattr(backend, "root_folder_id", None)
    if root_id:
        print(f"Google Drive root folder id: {root_id}")
    print(
        f"{action} complete: uploaded={result.uploaded}, "
        f"downloaded={result.downloaded}, unchanged={result.unchanged}, "
        f"remote_missing={result.remote_missing}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronize the VARnet-lite data lake.")
    parser.add_argument("action", choices=["pull", "push"])
    parser.add_argument("--backend", choices=["google_drive", "local"], default=None)
    parser.add_argument(
        "--if-configured",
        action="store_true",
        help="Exit successfully when Google Drive credentials are absent.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_sync(
            args.action,
            backend_name=args.backend,
            if_configured=args.if_configured,
        )
    )
