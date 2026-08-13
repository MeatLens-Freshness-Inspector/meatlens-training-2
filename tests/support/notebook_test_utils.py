from __future__ import annotations

import os
from pathlib import Path
from pprint import pformat

import nbformat


def execute_notebook(
    notebook_path: Path,
    overrides: dict[str, object] | None = None,
    cwd: Path | None = None,
):
    notebook = nbformat.read(notebook_path, as_version=4)
    test_overrides = {"DATASET_SOURCE": "current", **(overrides or {})}
    injected = nbformat.v4.new_code_cell(
        "NOTEBOOK_OVERRIDES = " + pformat(test_overrides, sort_dicts=True)
    )
    notebook.cells.insert(0, injected)
    execution_cwd = (cwd or notebook_path.parent).resolve()
    namespace: dict[str, object] = {
        "__name__": "__notebook__",
        "__file__": str(notebook_path.resolve()),
    }

    previous_cwd = Path.cwd()
    try:
        os.chdir(execution_cwd)
        for index, cell in enumerate(notebook.cells):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            code = compile(source, f"{notebook_path}::cell_{index}", "exec")
            exec(code, namespace)
    finally:
        os.chdir(previous_cwd)

    return namespace
