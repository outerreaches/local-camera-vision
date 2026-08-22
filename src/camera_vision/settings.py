from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from camera_vision.models import VisionSettings

DEFAULT_SETTINGS_PATH = Path("config/vision-config.yaml")


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> VisionSettings:
    config_path = Path(path)
    if not config_path.exists():
        return VisionSettings()
    with config_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] | None = yaml.safe_load(handle)
    return VisionSettings.model_validate(data or {})


def save_settings(settings: VisionSettings, path: str | Path = DEFAULT_SETTINGS_PATH) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(settings.model_dump(mode="json"), handle, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Camera Vision settings utility")
    parser.add_argument("--check", metavar="PATH", help="validate a YAML settings file")
    args = parser.parse_args(argv)
    if args.check:
        settings = load_settings(args.check)
        print(f"OK: {args.check} camera={settings.camera.name!r} port={settings.service.port}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
