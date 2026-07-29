from __future__ import annotations

from pathlib import Path


def test_windows_gpu_environment_is_pinned_for_tensorflow_210() -> None:
    content = Path("environment.windows-gpu.yml").read_text(encoding="utf-8")

    assert "name: meatlens-tf210-gpu" in content
    assert "cudatoolkit=11.2" in content
    assert "cudnn=8.1.0" in content
    assert "numpy<2" in content
    assert "tensorflow==2.10.1" in content
