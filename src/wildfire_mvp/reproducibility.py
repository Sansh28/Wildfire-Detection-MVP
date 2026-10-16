from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys


def utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def detect_git_commit(cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def command_log() -> dict[str, str | list[str]]:
    argv = list(sys.argv)
    rendered = " ".join(shlex.quote(item) for item in argv)
    return {
        "argv": argv,
        "command": rendered,
        "python_executable": sys.executable,
    }


def load_config_snapshot(config_path: str | Path) -> str:
    path = Path(config_path).resolve()
    return path.read_text(encoding="utf-8")


def save_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def collect_environment_info(extra_packages: list[str] | None = None) -> dict:
    packages = ["numpy", "pandas", "torch", "torchvision", "matplotlib", "rasterio", "requests"]
    if extra_packages:
        for package in extra_packages:
            if package not in packages:
                packages.append(package)

    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"

    accelerator_info = {
        "cuda_available": False,
        "mps_available": False,
        "cuda_device_count": 0,
    }
    try:
        import torch

        accelerator_info = {
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception:
        pass

    return {
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "accelerators": accelerator_info,
        "packages": versions,
    }


def select_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) is not None and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"
