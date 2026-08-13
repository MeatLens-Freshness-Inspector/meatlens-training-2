from __future__ import annotations

import json
from pathlib import Path

from meatlens_pork_pipeline.windows_tf_bootstrap import bootstrap_windows_tensorflow_dll_paths


def test_windows_gpu_environment_is_pinned_for_tensorflow_210() -> None:
    content = Path("environment.windows-gpu.yml").read_text(encoding="utf-8")

    assert "name: meatlens-tf210-gpu" in content
    assert "cudatoolkit=11.2" in content
    assert "cudnn=8.1.0" in content
    assert "numpy<2" in content
    assert "tensorflow==2.10.1" in content


def test_bootstrap_windows_tensorflow_dll_paths_prepends_conda_env_bins(tmp_path: Path) -> None:
    env_root = tmp_path / "envs" / "meatlens-tf210-gpu"
    python_executable = env_root / "python.exe"
    library_bin = env_root / "Library" / "bin"
    usr_bin = env_root / "Library" / "usr" / "bin"
    mingw_bin = env_root / "Library" / "mingw-w64" / "bin"
    scripts_dir = env_root / "Scripts"

    for path in (library_bin, usr_bin, mingw_bin, scripts_dir):
        path.mkdir(parents=True, exist_ok=True)
    python_executable.write_text("", encoding="utf-8")

    original_path = r"C:\Windows\System32"
    updated_path = bootstrap_windows_tensorflow_dll_paths(
        python_executable=python_executable,
        path_value=original_path,
        platform="win32",
    )

    expected_prefixes = [
        str(env_root),
        str(mingw_bin),
        str(usr_bin),
        str(library_bin),
        str(scripts_dir),
    ]
    assert updated_path.split(";")[:5] == expected_prefixes
    assert updated_path.endswith(original_path)


def test_shared_setup_bootstraps_windows_tensorflow_dlls_before_import() -> None:
    notebook = json.loads(Path("00_shared_setup.ipynb").read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "from meatlens_pork_pipeline.windows_tf_bootstrap import bootstrap_windows_tensorflow_dll_paths" in code_source
    assert "os.environ['PATH'] = bootstrap_windows_tensorflow_dll_paths(" in code_source
    assert code_source.index("bootstrap_windows_tensorflow_dll_paths") < code_source.index("import tensorflow as tensorflow_module")
