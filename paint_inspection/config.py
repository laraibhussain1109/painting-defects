from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass
class ProductProfile:
    name: str
    data: dict[str, Any]
    path: Path

    def section(self, name: str) -> dict[str, Any]:
        return self.data.setdefault(name, {})

    def save(self) -> None:
        self.data["name"] = self.name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")


class ProfileManager:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or ROOT / "config" / "products"
        self.directory.mkdir(parents=True, exist_ok=True)

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json"))

    def load(self, name: str) -> ProductProfile:
        path = self.directory / f"{name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return ProductProfile(str(data.get("name", name)), data, path)

    def create(self, name: str, source: str = "default") -> ProductProfile:
        safe = "".join(c for c in name if c.isalnum() or c in "_-").strip()
        if not safe:
            raise ValueError("Profile name must contain letters or numbers")
        target = self.directory / f"{safe}.json"
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(self.directory / f"{source}.json", target)
        profile = self.load(safe)
        profile.name = safe
        profile.save()
        return profile

    def copy(self, source: str, destination: str) -> ProductProfile:
        return self.create(destination, source)

    def delete(self, name: str) -> None:
        if name == "default":
            raise ValueError("The default profile cannot be deleted")
        (self.directory / f"{name}.json").unlink()


def load_yaml(relative: str) -> dict[str, Any]:
    with resolve_path(relative).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

