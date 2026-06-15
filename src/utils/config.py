from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(name: str) -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / name)


def ensure_dirs() -> None:
    for subdir in [
        DATA_DIR / "raw",
        DATA_DIR / "normalized",
        DATA_DIR / "features",
        DATA_DIR / "labels",
        DATA_DIR / "alternative",
        DATA_DIR / "models",
        DATA_DIR / "reports",
        DATA_DIR / "evaluation",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)
