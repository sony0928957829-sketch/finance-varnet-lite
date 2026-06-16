from __future__ import annotations

import io
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath

from .base import ObjectStorageBackend


FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


class GoogleDriveObjectStorage(ObjectStorageBackend):
    """Store data-lake objects below one app-owned Google Drive folder."""

    def __init__(
        self,
        *,
        credentials_payload: dict,
        root_folder_name: str,
        root_folder_id: str | None = None,
        scope: str = "https://www.googleapis.com/auth/drive.file",
        service=None,
    ):
        self.root_folder_name = root_folder_name
        self._folder_cache: dict[tuple[str, str], str] = {}
        self.service = service or self._build_service(credentials_payload, scope)
        self.root_folder_id = root_folder_id or self._find_or_create_root()

    @classmethod
    def from_environment(cls, config: dict) -> "GoogleDriveObjectStorage":
        credentials_env = config["credentials_json_env"]
        payload_text = os.environ.get(credentials_env, "").strip()
        if not payload_text:
            raise RuntimeError(
                f"{credentials_env} is not configured. Run the Google Drive "
                "authorization helper and store its JSON output as a secret."
            )
        root_id = os.environ.get(config.get("root_folder_id_env", ""), "").strip()
        return cls(
            credentials_payload=json.loads(payload_text),
            root_folder_name=config.get("root_folder_name", "VARnet-lite-data"),
            root_folder_id=root_id or None,
            scope=config.get(
                "oauth_scope",
                "https://www.googleapis.com/auth/drive.file",
            ),
        )

    @staticmethod
    def _build_service(payload: dict, scope: str):
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Google Drive dependencies are missing. Install requirements.txt."
            ) from exc

        credentials = Credentials(
            token=payload.get("token"),
            refresh_token=payload.get("refresh_token"),
            token_uri=payload.get(
                "token_uri",
                "https://oauth2.googleapis.com/token",
            ),
            client_id=payload.get("client_id"),
            client_secret=payload.get("client_secret"),
            scopes=payload.get("scopes") or [scope],
        )
        if not credentials.valid:
            credentials.refresh(Request())
        return build("drive", "v3", credentials=credentials, cache_discovery=False)

    @staticmethod
    def _escape_query(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _find_child(
        self,
        parent_id: str,
        name: str,
        *,
        mime_type: str | None = None,
    ) -> dict | None:
        clauses = [
            f"'{self._escape_query(parent_id)}' in parents",
            f"name = '{self._escape_query(name)}'",
            "trashed = false",
        ]
        if mime_type:
            clauses.append(f"mimeType = '{self._escape_query(mime_type)}'")
        response = (
            self.service.files()
            .list(
                q=" and ".join(clauses),
                spaces="drive",
                fields="files(id,name,mimeType,size,md5Checksum)",
                pageSize=10,
            )
            .execute()
        )
        files = response.get("files", [])
        return files[0] if files else None

    def _find_or_create_root(self) -> str:
        existing = self._find_child(
            "root",
            self.root_folder_name,
            mime_type=FOLDER_MIME_TYPE,
        )
        if existing:
            return existing["id"]
        created = (
            self.service.files()
            .create(
                body={
                    "name": self.root_folder_name,
                    "mimeType": FOLDER_MIME_TYPE,
                    "parents": ["root"],
                },
                fields="id",
            )
            .execute()
        )
        return created["id"]

    def _ensure_folder(self, parent_id: str, name: str) -> str:
        cache_key = (parent_id, name)
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]
        existing = self._find_child(parent_id, name, mime_type=FOLDER_MIME_TYPE)
        if existing:
            folder_id = existing["id"]
        else:
            folder_id = (
                self.service.files()
                .create(
                    body={
                        "name": name,
                        "mimeType": FOLDER_MIME_TYPE,
                        "parents": [parent_id],
                    },
                    fields="id",
                )
                .execute()["id"]
            )
        self._folder_cache[cache_key] = folder_id
        return folder_id

    def _resolve_parent(self, remote_path: str) -> tuple[str, str]:
        relative = PurePosixPath(remote_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.name:
            raise ValueError(f"Unsafe remote path: {remote_path}")
        parent_id = self.root_folder_id
        for part in relative.parts[:-1]:
            parent_id = self._ensure_folder(parent_id, part)
        return parent_id, relative.name

    def upload(self, local_path: Path, remote_path: str) -> None:
        from googleapiclient.http import MediaFileUpload

        parent_id, name = self._resolve_parent(remote_path)
        existing = self._find_child(parent_id, name)
        # Simple (non-resumable) upload: the data-lake objects are small
        # Parquet files, and resumable sessions add a per-file round-trip that
        # made the first full-history seed exceed the job time limit.
        media = MediaFileUpload(
            str(local_path),
            mimetype=mimetypes.guess_type(local_path.name)[0]
            or "application/octet-stream",
            resumable=False,
        )
        if existing:
            request = self.service.files().update(
                fileId=existing["id"],
                media_body=media,
                fields="id",
            )
        else:
            request = self.service.files().create(
                body={"name": name, "parents": [parent_id]},
                media_body=media,
                fields="id",
            )
        request.execute()

    def download(self, remote_path: str, local_path: Path) -> bool:
        from googleapiclient.http import MediaIoBaseDownload

        parent_id, name = self._resolve_parent(remote_path)
        existing = self._find_child(parent_id, name)
        if not existing:
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(
            buffer,
            self.service.files().get_media(fileId=existing["id"]),
        )
        done = False
        while not done:
            _, done = downloader.next_chunk()
        local_path.write_bytes(buffer.getvalue())
        return True
