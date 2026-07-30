from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap_windows_tensorflow_dll_paths(
    python_executable: str | Path | None = None,
    path_value: str | None = None,
    platform: str | None = None,
) -> str:
    current_platform = str(platform or sys.platform).lower()
    current_path = str(path_value if path_value is not None else os.environ.get("PATH", ""))

    if not current_platform.startswith("win"):
        return current_path

    executable_path = Path(python_executable or sys.executable).resolve()
    env_root = executable_path.parent if executable_path.name.lower().startswith("python") else executable_path

    candidate_dirs = [
        env_root,
        env_root / "Library" / "mingw-w64" / "bin",
        env_root / "Library" / "usr" / "bin",
        env_root / "Library" / "bin",
        env_root / "Scripts",
    ]
    existing_dirs = [str(path) for path in candidate_dirs if path.exists()]

    seen = {segment.lower() for segment in current_path.split(os.pathsep) if segment}
    prefixes = [segment for segment in existing_dirs if segment.lower() not in seen]
    updated_path = os.pathsep.join(prefixes + ([current_path] if current_path else []))

    if path_value is None:
        os.environ["PATH"] = updated_path
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if callable(add_dll_directory):
            for directory in existing_dirs:
                try:
                    add_dll_directory(directory)
                except (FileNotFoundError, OSError):
                    continue

    return updated_path


__all__ = ["bootstrap_windows_tensorflow_dll_paths"]
