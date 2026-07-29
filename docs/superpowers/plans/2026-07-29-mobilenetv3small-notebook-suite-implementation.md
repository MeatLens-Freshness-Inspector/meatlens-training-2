# MobileNetV3-Small Notebook Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a notebook-first MobileNetV3-small pork freshness training suite in the root workspace that mirrors the partner's processed-ROI 8-fold workflow and ends in a validated ONNX artifact.

**Architecture:** Move the operational workflow into a shared setup notebook plus multiple stage-specific notebooks at the repo root. Keep the existing Python package only as migration reference while the notebooks become the primary interface, the official evaluation stays in 8-fold cross-rotation notebooks, and deployment/export stay in separate notebooks.

**Tech Stack:** Python 3.10, JupyterLab, ipykernel, nbformat, nbclient, pandas, openpyxl, numpy, pillow, scipy, scikit-image, scikit-learn, tensorflow 2.16.1, matplotlib, seaborn, tf2onnx, onnx, onnxruntime, pytest

## Global Constraints

- The notebook suite is the primary interface; notebooks must live at the repo root and be runnable in order.
- The suite may use `%run ./00_shared_setup.ipynb` for shared code, but must not depend on the Python package as the main control surface.
- Official performance claims must come only from the `8-fold cross-rotation` workflow.
- Deployment notebooks produce the shippable artifact but do not replace the official evaluation.
- Backbone must be `MobileNetV3Small`.
- Model mode must be `cnn_only`.
- Input shape must be exactly `224x224x3`.
- Labels must be exactly `fresh`, `not fresh`, and `spoiled`.
- No handcrafted RGB, HSV, LAB, or GLCM feature branch may be used.
- Default seeds must be `42`, `123`, and `2026`.
- Default batch size must be `32`.
- Default head epochs must be `8`.
- Default fine-tune epochs must be `20`.
- Default head learning rate must be `5e-4`.
- Default fine-tune learning rate must be `1e-5`.
- Default fine-tune fraction must be the top `25%` of backbone layers.
- Training callbacks must monitor `val_f1_macro`.
- The canonical dataset root is `data/processed_hsv_lab_threshold_roi_224/`.
- The canonical manifest is `data/processed_hsv_lab_threshold_roi_224/processing_summary.csv`.
- The first notebook must default to the canonical processed-summary manifest and may additionally accept `.csv`, `.xlsx`, or `.xls` alternate manifests.
- Canonical processed-summary columns are `image_file_name`, `label`, `sample_number`, and `sample_id`.
- Preprocessing must remain `preprocessed_hsv_lab_threshold_roi_224`: center-square crop, resize to `224x224`, HSV/LAB-threshold ROI extraction, neutralized background, final RGB output.
- Processed images must be stored under `data/processed_hsv_lab_threshold_roi_224/`.
- The notebook suite must rebuild local processed-image paths from local columns and must not trust the `E:\...` paths stored in the CSV.
- The notebook suite must support mixed `.jpg` and `.png` processed images.
- Official split generation and reporting must tolerate unequal image counts across samples and labels.
- The notebook suite must output `accuracy`, macro `precision`, macro `recall`, macro `F1`, confusion matrix CSV, confusion matrix PNG, per-image predictions CSV, deployment metadata JSON, and ONNX model file.
- Intermediate saved Keras models should use native `.keras` format for Keras 3 compatibility; deployment output must still be ONNX.
- The official split notebook must stop unless it has a stable `sample_id` grouping suitable for the 8-fold cross-rotation workflow.

## Live Data Folder Adjustments

The repo already contains the processed dataset that the notebook suite should target first:

- `data/processed_hsv_lab_threshold_roi_224/processing_summary.csv` exists and should be the default manifest.
- `sample 1` through `sample 8` already exist as the official sample groups for cross-rotation.
- The dataset currently contains `4615` images:
  - `1415` `fresh`
  - `1600` `not fresh`
  - `1600` `spoiled`
- `fresh` counts are lower in sample `3` (`118`), sample `4` (`150`), and sample `7` (`147`), so no task may assume perfectly even fold cardinalities.
- Two processed images are `.png`, so image enumeration must not be JPG-only.
- The CSV's `processed_output_file` column points at an old `E:\...` location; implementation must rebuild local paths from `sample_number`, `label`, and `image_file_name`.

These live-data constraints supersede earlier raw-first assumptions in the lower-level examples below. Raw or Excel ingestion remains optional support work, not the default training path.

## Planned Files

**Create**

- `00_shared_setup.ipynb`
  - shared constants, paths, seed controls, helper functions, notebook test hooks
- `01_manifest_and_dataset_audit.ipynb`
  - canonical processed-summary audit, label normalization, local path resolution, dataset audits
- `02_roi_preprocessing.ipynb`
  - optional ROI preprocessing refresh and processed-manifest normalization
- `03_build_cross_rotation_splits.ipynb`
  - official 8-fold split generation and leakage checks
- `04_train_8fold_mobilenetv3small.ipynb`
  - official fold-and-seed training loop
- `05_regenerate_metrics_and_reports.ipynb`
  - aggregate metrics, confusion matrices, summary CSVs, figures
- `06_train_final_deployment_model.ipynb`
  - final all-samples deployment training
- `07_export_onnx.ipynb`
  - ONNX export and metadata generation
- `08_inference_smoke_test.ipynb`
  - ONNX Runtime sanity check
- `tests/notebook_test_utils.py`
  - notebook execution helper for pytest
- `tests/test_notebook_suite_structure.py`
  - shared setup notebook contract checks
- `tests/test_notebook_dataset_flow.py`
  - processed-summary audit + optional ROI preprocessing checks
- `tests/test_notebook_split_flow.py`
  - 8-fold split generation checks
- `tests/test_notebook_8fold_training.py`
  - official training notebook contract checks in test mode
- `tests/test_notebook_reports.py`
  - report regeneration checks
- `tests/test_notebook_final_deployment.py`
  - final deployment notebook checks
- `tests/test_notebook_onnx_workflow.py`
  - export + ONNX smoke workflow checks

**Modify**

- `environment.yml`
  - add Jupyter and notebook-testing dependencies
- `.gitignore`
  - ignore notebook checkpoints and execution debris
- `docs/training-pipeline-usage.md`
  - switch from script-first usage to notebook-first usage

### Task 1: Scaffold The Notebook Runtime Contract

**Files:**
- Modify: `environment.yml`
- Modify: `.gitignore`
- Create: `00_shared_setup.ipynb`
- Create: `tests/notebook_test_utils.py`
- Test: `tests/test_notebook_suite_structure.py`

**Interfaces:**
- Consumes: none
- Produces:
  - `override(name: str, default: object) -> object`
  - `NOTEBOOK_TEST_MODE: bool`
  - `ROOT: Path`
  - `DATA_ROOT: Path`
  - `RAW_DATA_ROOT: Path`
  - `GENERATED_SPLITS_ROOT: Path`
  - `PROCESSED_ROI_ROOT: Path`
  - `CANONICAL_PROCESSING_SUMMARY_PATH: Path`
  - `TRAINING_OUTPUTS_ROOT: Path`
  - `LABEL_ORDER: list[str]`
  - `RUN_SEEDS: list[int]`
  - `TARGET_SIZE: tuple[int, int]`
  - `INPUT_SHAPE: tuple[int, int, int]`
  - `BATCH_SIZE: int`
  - `EPOCHS_HEAD: int`
  - `EPOCHS_FINE: int`
  - `HEAD_LR: float`
  - `FINE_TUNE_LR: float`
  - `FINE_TUNE_FRACTION: float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_suite_structure.py
from __future__ import annotations

import json
from pathlib import Path


def test_shared_setup_notebook_exposes_core_contract() -> None:
    notebook = json.loads(Path("00_shared_setup.ipynb").read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "LABEL_ORDER = ['fresh', 'not fresh', 'spoiled']" in code_source
    assert "RUN_SEEDS = [42, 123, 2026]" in code_source
    assert "INPUT_SHAPE = (224, 224, 3)" in code_source
    assert "def override(name: str, default: object) -> object:" in code_source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_suite_structure.py::test_shared_setup_notebook_exposes_core_contract -v`

Expected: FAIL with `FileNotFoundError` for `00_shared_setup.ipynb`

- [ ] **Step 3: Write minimal implementation**

```yaml
# environment.yml
name: meatlens-pork-training
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - jupyterlab
  - ipykernel
  - nbformat
  - nbclient
  - numpy
  - pandas
  - openpyxl
  - pillow
  - scipy
  - scikit-image
  - scikit-learn
  - matplotlib
  - seaborn
  - pytest
  - pip:
      - tensorflow==2.16.1
      - tf2onnx==1.16.1
      - onnx==1.17.0
      - onnxruntime==1.19.2
```

```gitignore
# .gitignore
.ipynb_checkpoints/
__pycache__/
training_outputs/
processed_hsv_lab_threshold_roi_224/
```

```python
# tests/notebook_test_utils.py
from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_notebook(
    notebook_path: Path,
    overrides: dict[str, object] | None = None,
    cwd: Path | None = None,
):
    notebook = nbformat.read(notebook_path, as_version=4)
    injected = nbformat.v4.new_code_cell(
        "NOTEBOOK_OVERRIDES = " + json.dumps(overrides or {}, indent=2, sort_keys=True)
    )
    notebook.cells.insert(0, injected)
    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str((cwd or notebook_path.parent).resolve())}},
    )
    return client.execute()
```

```python
# 00_shared_setup.ipynb - code cells
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

NOTEBOOK_OVERRIDES = globals().get("NOTEBOOK_OVERRIDES", {})


def override(name: str, default: object) -> object:
    return NOTEBOOK_OVERRIDES.get(name, default)


NOTEBOOK_TEST_MODE = bool(override("NOTEBOOK_TEST_MODE", False))

ROOT = Path.cwd()
RAW_DATA_ROOT = Path(override("RAW_DATA_ROOT", ROOT / "raw_dataset"))
GENERATED_SPLITS_ROOT = ROOT / "generated_splits"
PROCESSED_ROI_ROOT = ROOT / "processed_hsv_lab_threshold_roi_224"
TRAINING_OUTPUTS_ROOT = ROOT / "training_outputs"

LABEL_ORDER = ['fresh', 'not fresh', 'spoiled']
RUN_SEEDS = [42, 123, 2026]
TARGET_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS_HEAD = 8
EPOCHS_FINE = 20
HEAD_LR = 5e-4
FINE_TUNE_LR = 1e-5
FINE_TUNE_FRACTION = 0.25


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


if override("WRITE_SETUP_CONTRACT", False):
    contract_path = Path(str(override("SETUP_CONTRACT_OUT", ROOT / "setup_contract.json")))
    ensure_dir(contract_path.parent)
    contract = {
        "label_order": LABEL_ORDER,
        "run_seeds": RUN_SEEDS,
        "input_shape": list(INPUT_SHAPE),
        "batch_size": BATCH_SIZE,
    }
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(contract_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_suite_structure.py -v`

Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add environment.yml .gitignore 00_shared_setup.ipynb tests/notebook_test_utils.py tests/test_notebook_suite_structure.py
git commit -m "chore: scaffold notebook runtime contract"
```

### Task 2: Build The Manifest Audit And ROI Preprocessing Notebooks

**Files:**
- Modify: `00_shared_setup.ipynb`
- Create: `01_manifest_and_dataset_audit.ipynb`
- Create: `02_roi_preprocessing.ipynb`
- Test: `tests/test_notebook_dataset_flow.py`

**Interfaces:**
- Consumes:
  - `override(name: str, default: object) -> object`
  - `ensure_dir(path: Path) -> Path`
  - `LABEL_ORDER: list[str]`
  - `TARGET_SIZE: tuple[int, int]`
- Produces:
  - `normalize_label(value: str) -> str`
  - `load_manifest_table(manifest_path: Path) -> pd.DataFrame`
  - `resolve_manifest_image_paths(manifest_df: pd.DataFrame, images_root: Path) -> pd.DataFrame`
  - `preprocess_roi_image(path: Path, background_mode: str = "gray") -> tuple[np.ndarray, dict[str, object]]`
  - `preprocess_manifest_images(manifest_df: pd.DataFrame, output_root: Path, background_mode: str = "gray") -> tuple[pd.DataFrame, Path, Path]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_dataset_flow.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from tests.notebook_test_utils import execute_notebook


def test_manifest_audit_and_roi_preprocessing_create_processed_dataset(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    images_root.mkdir()

    rows = []
    for idx, label in enumerate(["Fresh", "not_fresh", "spoiled"], start=1):
        image_name = f"sample_{idx}_{label}.jpg".replace(" ", "_")
        Image.new("RGB", (320, 280), (150 + idx * 10, 50, 50)).save(images_root / image_name)
        rows.append({"image_name": image_name, "label": label, "sample_id": f"sample_{idx}"})

    manifest_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    audited_manifest_path = tmp_path / "audited_manifest.csv"
    processed_manifest_path = tmp_path / "processed_manifest.csv"
    preprocessing_summary_path = tmp_path / "preprocessing_summary.csv"
    preprocessing_failures_path = tmp_path / "preprocessing_failures.csv"

    execute_notebook(
        Path("01_manifest_and_dataset_audit.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "MANIFEST_PATH": str(manifest_path),
            "IMAGES_ROOT": str(images_root),
            "AUDITED_MANIFEST_PATH": str(audited_manifest_path),
        },
    )
    audited_df = pd.read_csv(audited_manifest_path)
    assert list(audited_df["label"]) == ["fresh", "not fresh", "spoiled"]
    assert audited_df["image_path_resolved"].map(Path).map(Path.exists).all()

    execute_notebook(
        Path("02_roi_preprocessing.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "AUDITED_MANIFEST_PATH": str(audited_manifest_path),
            "PROCESSED_MANIFEST_PATH": str(processed_manifest_path),
            "PREPROCESSING_SUMMARY_PATH": str(preprocessing_summary_path),
            "PREPROCESSING_FAILURES_PATH": str(preprocessing_failures_path),
            "PROCESSED_ROI_ROOT": str(tmp_path / "processed_hsv_lab_threshold_roi_224"),
        },
    )
    processed_df = pd.read_csv(processed_manifest_path)
    processed_image = Path(processed_df.loc[0, "processed_image_path"])
    assert processed_image.exists()
    assert Image.open(processed_image).size == (224, 224)
    assert preprocessing_summary_path.exists()
    assert preprocessing_failures_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_dataset_flow.py::test_manifest_audit_and_roi_preprocessing_create_processed_dataset -v`

Expected: FAIL with `FileNotFoundError` for `01_manifest_and_dataset_audit.ipynb`

- [ ] **Step 3: Write minimal implementation**

```python
# 00_shared_setup.ipynb - add code cells
import re
from typing import Iterable

import numpy as np
from PIL import Image

from apply_hsv_lab_threshold_roi_batch import (
    apply_background_fill,
    clean_mask,
    failure_flag,
    get_hsv_channels,
    preprocess_center_square_resize_224,
)


def normalize_label(value: str) -> str:
    key = value.strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    mapping = {
        "fresh": "fresh",
        "not fresh": "not fresh",
        "spoiled": "spoiled",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported label: {value}")
    return mapping[key]


def load_manifest_table(manifest_path: Path) -> pd.DataFrame:
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(manifest_path, dtype=str).fillna("")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(manifest_path, dtype=str).fillna("")
    else:
        raise ValueError(f"Unsupported manifest file: {manifest_path}")
    required = {"image_name", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def resolve_manifest_image_paths(manifest_df: pd.DataFrame, images_root: Path) -> pd.DataFrame:
    def resolve_one(image_name: str) -> str:
        matches = sorted(path for path in images_root.rglob(image_name) if path.is_file())
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one match for {image_name}, found {len(matches)}")
        return str(matches[0])

    df = manifest_df.copy()
    df["label"] = df["label"].map(normalize_label)
    df["image_path_resolved"] = df["image_name"].astype(str).map(resolve_one)
    if "sample_id" not in df.columns:
        df["sample_id"] = ""
    return df


def _lab_channels_numpy(image_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = image_uint8.astype(np.float32) / 255.0
    rgb_linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    x = ((rgb_linear[:, :, 0] * 0.4124564) + (rgb_linear[:, :, 1] * 0.3575761) + (rgb_linear[:, :, 2] * 0.1804375)) / 0.95047
    y = ((rgb_linear[:, :, 0] * 0.2126729) + (rgb_linear[:, :, 1] * 0.7151522) + (rgb_linear[:, :, 2] * 0.0721750)) / 1.00000
    z = ((rgb_linear[:, :, 0] * 0.0193339) + (rgb_linear[:, :, 1] * 0.1191920) + (rgb_linear[:, :, 2] * 0.9503041)) / 1.08883
    delta = 6.0 / 29.0
    delta_cubed = delta**3
    scale = 1.0 / (3.0 * delta**2)
    offset = 4.0 / 29.0

    def transform(channel: np.ndarray) -> np.ndarray:
        return np.where(channel > delta_cubed, np.cbrt(channel), (channel * scale) + offset)

    fx = transform(x)
    fy = transform(y)
    fz = transform(z)
    l = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return l.astype(np.float32), a.astype(np.float32), b.astype(np.float32)


def preprocess_roi_image(path: Path, background_mode: str = "gray") -> tuple[np.ndarray, dict[str, object]]:
    image = Image.open(path).convert("RGB")
    square = preprocess_center_square_resize_224(image)
    image_uint8 = np.asarray(square, dtype=np.uint8)
    _, saturation, value = get_hsv_channels(image_uint8)
    _, a_channel, _ = _lab_channels_numpy(image_uint8)
    raw_mask = (
        (saturation >= 20)
        & (value >= 35)
        & ~((value >= 235) & (saturation <= 30))
        & (a_channel >= 6.0)
    )
    final_mask, quality = clean_mask(raw_mask)
    failed = failure_flag(final_mask, quality)
    output_uint8 = image_uint8.copy() if failed or not final_mask.any() else apply_background_fill(image_uint8, final_mask, mode=background_mode)
    metadata = {
        "segmentation_failed": failed,
        "mask_area_ratio": float(quality["mask_area_ratio"]),
        "center_overlap_ratio": float(quality["center_overlap_ratio"]),
        "number_of_components": int(quality["number_of_components"]),
        "touches_border": bool(quality["touches_border"]),
    }
    return output_uint8, metadata


def preprocess_manifest_images(
    manifest_df: pd.DataFrame,
    output_root: Path,
    background_mode: str = "gray",
) -> tuple[pd.DataFrame, Path, Path]:
    output_root = ensure_dir(output_root)
    summary_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []

    for row in manifest_df.to_dict(orient="records"):
        source_path = Path(str(row["image_path_resolved"]))
        label = str(row["label"])
        output_path = output_root / label / source_path.name
        ensure_dir(output_path.parent)
        try:
            processed_uint8, metadata = preprocess_roi_image(source_path, background_mode=background_mode)
            Image.fromarray(processed_uint8).save(output_path, quality=95)
            output_rows.append({**row, "processed_image_path": str(output_path)})
            summary_rows.append({"image_name": source_path.name, "label": label, "processed_image_path": str(output_path), **metadata})
        except Exception as exc:
            failure_rows.append({"image_name": source_path.name, "label": label, "error": repr(exc)})

    processed_df = pd.DataFrame(output_rows)
    summary_path = output_root / "preprocessing_summary.csv"
    failures_path = output_root / "preprocessing_failures.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(failure_rows).to_csv(failures_path, index=False)
    return processed_df, summary_path, failures_path
```

```python
# 01_manifest_and_dataset_audit.ipynb - code cells
%run ./00_shared_setup.ipynb

MANIFEST_PATH = Path(str(override("MANIFEST_PATH", RAW_DATA_ROOT / "labels.xlsx")))
IMAGES_ROOT = Path(str(override("IMAGES_ROOT", RAW_DATA_ROOT / "images")))
AUDITED_MANIFEST_PATH = Path(str(override("AUDITED_MANIFEST_PATH", GENERATED_SPLITS_ROOT / "audited_manifest.csv")))

manifest_df = load_manifest_table(MANIFEST_PATH)
audited_df = resolve_manifest_image_paths(manifest_df, IMAGES_ROOT)
ensure_dir(AUDITED_MANIFEST_PATH.parent)
audited_df.to_csv(AUDITED_MANIFEST_PATH, index=False)
display(audited_df.head())
print(AUDITED_MANIFEST_PATH)
```

```python
# 02_roi_preprocessing.ipynb - code cells
%run ./00_shared_setup.ipynb

AUDITED_MANIFEST_PATH = Path(str(override("AUDITED_MANIFEST_PATH", GENERATED_SPLITS_ROOT / "audited_manifest.csv")))
PROCESSED_MANIFEST_PATH = Path(str(override("PROCESSED_MANIFEST_PATH", GENERATED_SPLITS_ROOT / "processed_manifest.csv")))
PREPROCESSING_SUMMARY_PATH = Path(str(override("PREPROCESSING_SUMMARY_PATH", PROCESSED_ROI_ROOT / "preprocessing_summary.csv")))
PREPROCESSING_FAILURES_PATH = Path(str(override("PREPROCESSING_FAILURES_PATH", PROCESSED_ROI_ROOT / "preprocessing_failures.csv")))
BACKGROUND_MODE = str(override("BACKGROUND_MODE", "gray"))

audited_df = pd.read_csv(AUDITED_MANIFEST_PATH)
processed_df, summary_path, failures_path = preprocess_manifest_images(
    audited_df,
    output_root=PROCESSED_ROI_ROOT,
    background_mode=BACKGROUND_MODE,
)
ensure_dir(PROCESSED_MANIFEST_PATH.parent)
processed_df.to_csv(PROCESSED_MANIFEST_PATH, index=False)
Path(summary_path).replace(PREPROCESSING_SUMMARY_PATH)
Path(failures_path).replace(PREPROCESSING_FAILURES_PATH)
display(processed_df.head())
print(PROCESSED_MANIFEST_PATH)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_dataset_flow.py -v`

Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add 00_shared_setup.ipynb 01_manifest_and_dataset_audit.ipynb 02_roi_preprocessing.ipynb tests/test_notebook_dataset_flow.py
git commit -m "feat: add notebook dataset audit and ROI preprocessing"
```

### Task 3: Build The Official Cross-Rotation Split Notebook

**Files:**
- Modify: `00_shared_setup.ipynb`
- Create: `03_build_cross_rotation_splits.ipynb`
- Test: `tests/test_notebook_split_flow.py`

**Interfaces:**
- Consumes:
  - `ensure_dir(path: Path) -> Path`
  - `override(name: str, default: object) -> object`
  - `LABEL_ORDER: list[str]`
  - `PROCESSED_ROI_ROOT: Path`
- Produces:
  - `require_official_sample_ids(manifest_df: pd.DataFrame, expected_count: int = 8) -> pd.DataFrame`
  - `build_official_cross_rotation_splits(processed_df: pd.DataFrame, output_root: Path) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_split_flow.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.notebook_test_utils import execute_notebook


def test_cross_rotation_split_notebook_writes_fold_csvs_without_leakage(tmp_path: Path) -> None:
    rows = []
    for sample_idx in range(1, 9):
        for label in ["fresh", "not fresh", "spoiled"]:
            for image_idx in range(2):
                image_name = f"sample_{sample_idx}_{label}_{image_idx}.jpg".replace(" ", "_")
                processed_path = tmp_path / "processed_hsv_lab_threshold_roi_224" / label / image_name
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                processed_path.write_bytes(b"fake")
                rows.append(
                    {
                        "image_name": image_name,
                        "label": label,
                        "sample_id": f"sample_{sample_idx}",
                        "processed_image_path": str(processed_path),
                    }
                )

    processed_manifest_path = tmp_path / "processed_manifest.csv"
    pd.DataFrame(rows).to_csv(processed_manifest_path, index=False)

    output_root = tmp_path / "generated_splits" / "cross_rotation_interval200_8samples_processed_roi"
    execute_notebook(
        Path("03_build_cross_rotation_splits.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "PROCESSED_MANIFEST_PATH": str(processed_manifest_path),
            "CROSS_ROTATION_OUTPUT_ROOT": str(output_root),
        },
    )

    assert (output_root / "fold1_train.csv").exists()
    assert (output_root / "fold1_val.csv").exists()
    assert (output_root / "fold1_test.csv").exists()
    leakage_df = pd.read_csv(output_root / "cross_rotation_leakage_check.csv")
    assert leakage_df["has_leakage"].eq(False).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_split_flow.py::test_cross_rotation_split_notebook_writes_fold_csvs_without_leakage -v`

Expected: FAIL with `FileNotFoundError` for `03_build_cross_rotation_splits.ipynb`

- [ ] **Step 3: Write minimal implementation**

```python
# 00_shared_setup.ipynb - add code cells
def require_official_sample_ids(manifest_df: pd.DataFrame, expected_count: int = 8) -> pd.DataFrame:
    df = manifest_df.copy()
    if "sample_id" not in df.columns:
        raise ValueError("Official 8-fold cross-rotation requires a sample_id column.")
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    if (df["sample_id"] == "").any():
        raise ValueError("Official 8-fold cross-rotation requires non-empty sample_id values.")
    unique_sample_ids = sorted(df["sample_id"].unique().tolist())
    if len(unique_sample_ids) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} unique sample_id values, found {len(unique_sample_ids)}")
    return df


def build_official_cross_rotation_splits(
    processed_df: pd.DataFrame,
    output_root: Path,
) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]:
    output_root = ensure_dir(output_root)
    unique_sample_ids = sorted(processed_df["sample_id"].astype(str).unique().tolist())
    split_paths: dict[str, Path] = {}
    summary_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []

    all_images_path = output_root / "all_sampled_images.csv"
    processed_df.to_csv(all_images_path, index=False)

    for fold_index, test_sample in enumerate(unique_sample_ids, start=1):
        val_sample = unique_sample_ids[fold_index % len(unique_sample_ids)]
        fold_name = f"fold{fold_index}"
        test_df = processed_df.loc[processed_df["sample_id"].eq(test_sample)].copy()
        val_df = processed_df.loc[processed_df["sample_id"].eq(val_sample)].copy()
        train_df = processed_df.loc[
            ~processed_df["sample_id"].isin([test_sample, val_sample])
        ].copy()

        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            split_path = output_root / f"{fold_name}_{split_name}.csv"
            split_df.to_csv(split_path, index=False)
            split_paths[f"{fold_name}_{split_name}"] = split_path
            summary_rows.append(
                {
                    "fold_name": fold_name,
                    "split_name": split_name,
                    "count": int(len(split_df)),
                    "sample_ids": ",".join(sorted(split_df["sample_id"].astype(str).unique().tolist())),
                }
            )

        leakage_rows.append(
            {
                "fold_name": fold_name,
                "test_sample_id": test_sample,
                "val_sample_id": val_sample,
                "has_leakage": bool(
                    set(train_df["sample_id"]).intersection(test_df["sample_id"])
                    or set(train_df["sample_id"]).intersection(val_df["sample_id"])
                    or set(test_df["sample_id"]).intersection(val_df["sample_id"])
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    leakage_df = pd.DataFrame(leakage_rows)
    summary_df.to_csv(output_root / "cross_rotation_summary.csv", index=False)
    leakage_df.to_csv(output_root / "cross_rotation_leakage_check.csv", index=False)
    return split_paths, summary_df, leakage_df
```

```python
# 03_build_cross_rotation_splits.ipynb - code cells
%run ./00_shared_setup.ipynb

PROCESSED_MANIFEST_PATH = Path(str(override("PROCESSED_MANIFEST_PATH", GENERATED_SPLITS_ROOT / "processed_manifest.csv")))
CROSS_ROTATION_OUTPUT_ROOT = Path(
    str(
        override(
            "CROSS_ROTATION_OUTPUT_ROOT",
            GENERATED_SPLITS_ROOT / "cross_rotation_interval200_8samples_processed_roi",
        )
    )
)

processed_df = pd.read_csv(PROCESSED_MANIFEST_PATH)
processed_df = require_official_sample_ids(processed_df, expected_count=8)
split_paths, summary_df, leakage_df = build_official_cross_rotation_splits(processed_df, CROSS_ROTATION_OUTPUT_ROOT)
display(summary_df.head())
display(leakage_df)
print(CROSS_ROTATION_OUTPUT_ROOT)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_split_flow.py -v`

Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add 00_shared_setup.ipynb 03_build_cross_rotation_splits.ipynb tests/test_notebook_split_flow.py
git commit -m "feat: add cross-rotation split notebook"
```

### Task 4: Build The Official 8-Fold Training Notebook

**Files:**
- Modify: `00_shared_setup.ipynb`
- Create: `04_train_8fold_mobilenetv3small.ipynb`
- Test: `tests/test_notebook_8fold_training.py`

**Interfaces:**
- Consumes:
  - `build_official_cross_rotation_splits(...) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]`
  - `require_official_sample_ids(...) -> pd.DataFrame`
  - `LABEL_ORDER: list[str]`
  - `RUN_SEEDS: list[int]`
  - `BATCH_SIZE: int`
  - `EPOCHS_HEAD: int`
  - `EPOCHS_FINE: int`
- Produces:
  - `ImageOnlySequence`
  - `build_mobilenetv3small_cnn_only_model(weights: str | None = "imagenet") -> tf.keras.Model`
  - `compute_class_weights_from_labels(labels: np.ndarray) -> dict[int, float]`
  - `train_single_fold(fold_name: str, seed: int, train_csv: Path, val_csv: Path, test_csv: Path, output_root: Path) -> dict[str, object]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_8fold_training.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from tests.notebook_test_utils import execute_notebook


def test_8fold_training_notebook_writes_metrics_predictions_and_model(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed_hsv_lab_threshold_roi_224"
    rows = []
    for sample_idx in range(1, 9):
        for label_idx, label in enumerate(["fresh", "not fresh", "spoiled"]):
            for image_idx in range(2):
                image_name = f"sample_{sample_idx}_{label}_{image_idx}.jpg".replace(" ", "_")
                image_path = processed_root / label / image_name
                image_path.parent.mkdir(parents=True, exist_ok=True)
                array = np.full((224, 224, 3), 50 + sample_idx * 10 + label_idx * 20, dtype=np.uint8)
                Image.fromarray(array).save(image_path)
                rows.append(
                    {
                        "image_name": image_name,
                        "label": label,
                        "sample_id": f"sample_{sample_idx}",
                        "processed_image_path": str(image_path),
                    }
                )

    processed_manifest_path = tmp_path / "processed_manifest.csv"
    pd.DataFrame(rows).to_csv(processed_manifest_path, index=False)

    split_root = tmp_path / "generated_splits" / "cross_rotation_interval200_8samples_processed_roi"
    execute_notebook(
        Path("03_build_cross_rotation_splits.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "PROCESSED_MANIFEST_PATH": str(processed_manifest_path),
            "CROSS_ROTATION_OUTPUT_ROOT": str(split_root),
        },
    )

    output_root = tmp_path / "training_outputs" / "mobilenetv3small_8fold_processed_roi_cnn_only"
    execute_notebook(
        Path("04_train_8fold_mobilenetv3small.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "CROSS_ROTATION_OUTPUT_ROOT": str(split_root),
            "TRAINING_OUTPUT_ROOT": str(output_root),
            "RUN_SEEDS": [42],
            "SELECT_FOLDS": ["fold1"],
            "MODEL_WEIGHTS": None,
            "BATCH_SIZE": 4,
            "EPOCHS_HEAD": 1,
            "EPOCHS_FINE": 1,
            "USE_TRAINING_AUGMENTATION": False,
        },
    )

    assert (output_root / "processed_roi8_cnn_only_seed_metrics.csv").exists()
    assert (output_root / "models" / "processed_roi8_cnn_only_fold1_seed42.keras").exists()
    assert (output_root / "predictions" / "processed_roi8_cnn_only_fold1_seed42_predictions.csv").exists()
    assert (output_root / "figures" / "processed_roi8_cnn_only_fold1_seed42_confusion_matrix.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_8fold_training.py::test_8fold_training_notebook_writes_metrics_predictions_and_model -v`

Expected: FAIL with `FileNotFoundError` for `04_train_8fold_mobilenetv3small.ipynb`

- [ ] **Step 3: Write minimal implementation**

```python
# 00_shared_setup.ipynb - add code cells
import json
import math
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.applications import MobileNetV3Small


def preprocess_mobilenetv3_batch(batch_uint8: np.ndarray) -> np.ndarray:
    batch = batch_uint8.astype(np.float32)
    return ((batch / 127.5) - 1.0).astype(np.float32)


def augment_image_uint8(image_uint8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image = Image.fromarray(image_uint8)
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return np.asarray(image, dtype=np.uint8)


class ImageOnlySequence(tf.keras.utils.Sequence):
    def __init__(self, image_paths: list[str], labels: np.ndarray | None, batch_size: int, shuffle: bool, augment: bool):
        self.image_paths = list(image_paths)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int32)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / self.batch_size)) if len(self.indices) else 0

    def __getitem__(self, index: int):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        batch_images = []
        for idx in batch_indices:
            image_uint8 = np.asarray(Image.open(self.image_paths[idx]).convert("RGB"), dtype=np.uint8)
            if self.augment:
                image_uint8 = augment_image_uint8(image_uint8, np.random.default_rng(np.random.randint(0, 2**31 - 1)))
            batch_images.append(preprocess_mobilenetv3_batch(image_uint8[None, ...])[0])
        batch = np.stack(batch_images).astype(np.float32)
        if self.labels is None:
            return batch
        return batch, self.labels[batch_indices]

    def on_epoch_end(self) -> None:
        if self.shuffle and len(self.indices) > 0:
            np.random.shuffle(self.indices)


def build_mobilenetv3small_cnn_only_model(weights: str | None = "imagenet") -> tf.keras.Model:
    image_input = layers.Input(shape=INPUT_SHAPE, name="image_input")
    backbone = MobileNetV3Small(include_top=False, weights=weights, input_shape=INPUT_SHAPE)
    backbone._name = "mobilenetv3small_backbone"
    backbone.trainable = False
    x = backbone(image_input, training=False)
    x = layers.GlobalAveragePooling2D(name="image_gap")(x)
    x = layers.Dropout(0.30, name="image_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.30, name="image_dropout_2")(x)
    output = layers.Dense(len(LABEL_ORDER), activation="softmax", name="classification_head")(x)
    return models.Model(inputs=image_input, outputs=output, name="meatlens_processed_roi8_cnn_only")


def compute_class_weights_from_labels(labels: np.ndarray) -> dict[int, float]:
    classes = np.array([0, 1, 2], dtype=np.int32)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(labels, dtype=np.int32))
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


class ValMacroF1Callback(callbacks.Callback):
    def __init__(self, val_source, val_labels):
        super().__init__()
        self.val_source = val_source
        self.val_labels = np.asarray(val_labels, dtype=np.int32)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_prob = self.model.predict(self.val_source, verbose=0)
        y_pred = y_prob.argmax(axis=1)
        _, _, f1, _ = precision_recall_fscore_support(self.val_labels, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        logs["val_f1_macro"] = float(f1)
        print(f" - val_f1_macro: {float(f1):.4f}")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix_json": json.dumps(cm.tolist()),
    }


def unfreeze_top_backbone_fraction(model: tf.keras.Model, fraction: float) -> None:
    backbone = model.get_layer("mobilenetv3small_backbone")
    backbone.trainable = True
    cutoff = max(int(len(backbone.layers) * (1.0 - fraction)), 0)
    for layer in backbone.layers[:cutoff]:
        layer.trainable = False


def save_confusion_matrix_png(matrix: np.ndarray, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    ensure_dir(output_path.parent)
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def train_single_fold(
    fold_name: str,
    seed: int,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    output_root: Path,
) -> dict[str, object]:
    set_global_seed(seed)
    output_root = ensure_dir(output_root)
    models_root = ensure_dir(output_root / "models")
    predictions_root = ensure_dir(output_root / "predictions")
    figures_root = ensure_dir(output_root / "figures")
    logs_root = ensure_dir(output_root / "logs")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_labels = train_df["label"].map({name: idx for idx, name in enumerate(LABEL_ORDER)}).astype(int).to_numpy()
    val_labels = val_df["label"].map({name: idx for idx, name in enumerate(LABEL_ORDER)}).astype(int).to_numpy()
    test_labels = test_df["label"].map({name: idx for idx, name in enumerate(LABEL_ORDER)}).astype(int).to_numpy()

    train_sequence = ImageOnlySequence(train_df["processed_image_path"].astype(str).tolist(), train_labels, int(override("BATCH_SIZE", BATCH_SIZE)), True, bool(override("USE_TRAINING_AUGMENTATION", True)))
    val_sequence = ImageOnlySequence(val_df["processed_image_path"].astype(str).tolist(), val_labels, int(override("BATCH_SIZE", BATCH_SIZE)), False, False)
    test_sequence = ImageOnlySequence(test_df["processed_image_path"].astype(str).tolist(), None, int(override("BATCH_SIZE", BATCH_SIZE)), False, False)

    model = build_mobilenetv3small_cnn_only_model(weights=override("MODEL_WEIGHTS", "imagenet"))
    class_weights = compute_class_weights_from_labels(train_labels)
    run_stem = f"processed_roi8_cnn_only_{fold_name}_seed{seed}"

    cb = [
        ValMacroF1Callback(val_sequence, val_labels),
        callbacks.EarlyStopping(monitor="val_f1_macro", mode="max", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_f1_macro", mode="max", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]

    model.compile(optimizer=tf.keras.optimizers.Adam(float(override("HEAD_LR", HEAD_LR))), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history_head = model.fit(train_sequence, validation_data=val_sequence, epochs=int(override("EPOCHS_HEAD", EPOCHS_HEAD)), class_weight=class_weights, callbacks=cb, verbose=2)

    unfreeze_top_backbone_fraction(model, float(override("FINE_TUNE_FRACTION", FINE_TUNE_FRACTION)))
    model.compile(optimizer=tf.keras.optimizers.Adam(float(override("FINE_TUNE_LR", FINE_TUNE_LR))), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history_fine = model.fit(train_sequence, validation_data=val_sequence, epochs=int(override("EPOCHS_FINE", EPOCHS_FINE)), class_weight=class_weights, callbacks=cb, verbose=2)

    y_prob = model.predict(test_sequence, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    metric_row = classification_metrics(test_labels, y_pred)

    prediction_df = test_df.copy()
    prediction_df["true_label"] = test_df["label"]
    prediction_df["predicted_label"] = [LABEL_ORDER[index] for index in y_pred]
    prediction_df["confidence"] = y_prob.max(axis=1)
    prediction_path = predictions_root / f"{run_stem}_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False)

    model_path = models_root / f"{run_stem}.keras"
    model.save(model_path, include_optimizer=False)

    save_confusion_matrix_png(np.array(json.loads(metric_row["confusion_matrix_json"])), figures_root / f"{run_stem}_confusion_matrix.png", f"{run_stem} Confusion Matrix")
    pd.DataFrame({"phase": ["head", "fine"], "epochs": [len(history_head.epoch), len(history_fine.epoch)]}).to_csv(logs_root / f"{run_stem}_history.csv", index=False)

    return {
        "fold_name": fold_name,
        "seed": seed,
        "model_input_mode": "cnn_only",
        "image_crop_mode": "preprocessed_hsv_lab_threshold_roi_224",
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "test_count": int(len(test_df)),
        "predictions_path": str(prediction_path),
        "model_path": str(model_path),
        "class_weights_json": json.dumps(class_weights),
        **metric_row,
    }
```

```python
# 04_train_8fold_mobilenetv3small.ipynb - code cells
%run ./00_shared_setup.ipynb

CROSS_ROTATION_OUTPUT_ROOT = Path(
    str(
        override(
            "CROSS_ROTATION_OUTPUT_ROOT",
            GENERATED_SPLITS_ROOT / "cross_rotation_interval200_8samples_processed_roi",
        )
    )
)
TRAINING_OUTPUT_ROOT = Path(
    str(
        override(
            "TRAINING_OUTPUT_ROOT",
            TRAINING_OUTPUTS_ROOT / "mobilenetv3small_8fold_processed_roi_cnn_only",
        )
    )
)
SELECT_FOLDS = list(override("SELECT_FOLDS", [f"fold{i}" for i in range(1, 9)]))
ACTIVE_SEEDS = list(override("RUN_SEEDS", RUN_SEEDS))

seed_results = []
for fold_name in SELECT_FOLDS:
    for seed in ACTIVE_SEEDS:
        result = train_single_fold(
            fold_name=fold_name,
            seed=int(seed),
            train_csv=CROSS_ROTATION_OUTPUT_ROOT / f"{fold_name}_train.csv",
            val_csv=CROSS_ROTATION_OUTPUT_ROOT / f"{fold_name}_val.csv",
            test_csv=CROSS_ROTATION_OUTPUT_ROOT / f"{fold_name}_test.csv",
            output_root=TRAINING_OUTPUT_ROOT,
        )
        seed_results.append(result)

seed_metrics_df = pd.DataFrame(seed_results).sort_values(["fold_name", "seed"]).reset_index(drop=True)
ensure_dir(TRAINING_OUTPUT_ROOT)
seed_metrics_df.to_csv(TRAINING_OUTPUT_ROOT / "processed_roi8_cnn_only_seed_metrics.csv", index=False)
display(seed_metrics_df.head())
print(TRAINING_OUTPUT_ROOT)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_8fold_training.py -v`

Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add 00_shared_setup.ipynb 04_train_8fold_mobilenetv3small.ipynb tests/test_notebook_8fold_training.py
git commit -m "feat: add official 8-fold training notebook"
```

### Task 5: Build The Metrics And Reports Regeneration Notebook

**Files:**
- Modify: `00_shared_setup.ipynb`
- Create: `05_regenerate_metrics_and_reports.ipynb`
- Test: `tests/test_notebook_reports.py`

**Interfaces:**
- Consumes:
  - `classification_metrics(...) -> dict[str, object]`
  - `save_confusion_matrix_png(matrix: np.ndarray, output_path: Path, title: str) -> None`
  - `LABEL_ORDER: list[str]`
- Produces:
  - `load_prediction_csvs_for_metrics(seed_metrics_path: Path) -> pd.DataFrame`
  - `regenerate_official_reports(seed_metrics_path: Path, output_root: Path) -> dict[str, Path]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_reports.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tests.notebook_test_utils import execute_notebook


def test_report_notebook_rebuilds_summary_outputs_from_saved_predictions(tmp_path: Path) -> None:
    output_root = tmp_path / "training_outputs" / "mobilenetv3small_8fold_processed_roi_cnn_only"
    predictions_root = output_root / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)

    prediction_df = pd.DataFrame(
        [
            {"true_label": "fresh", "predicted_label": "fresh"},
            {"true_label": "fresh", "predicted_label": "not fresh"},
            {"true_label": "not fresh", "predicted_label": "not fresh"},
            {"true_label": "spoiled", "predicted_label": "spoiled"},
        ]
    )
    prediction_path = predictions_root / "processed_roi8_cnn_only_fold1_seed42_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False)

    seed_metrics_df = pd.DataFrame(
        [
            {
                "fold_name": "fold1",
                "seed": 42,
                "model_input_mode": "cnn_only",
                "image_crop_mode": "preprocessed_hsv_lab_threshold_roi_224",
                "predictions_path": str(prediction_path),
                "accuracy": 0.75,
                "macro_precision": 0.8333,
                "macro_recall": 0.8333,
                "macro_f1": 0.7777,
            }
        ]
    )
    seed_metrics_path = output_root / "processed_roi8_cnn_only_seed_metrics.csv"
    seed_metrics_df.to_csv(seed_metrics_path, index=False)

    execute_notebook(
        Path("05_regenerate_metrics_and_reports.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "TRAINING_OUTPUT_ROOT": str(output_root),
        },
    )

    assert (output_root / "processed_roi8_cnn_only_fold_summary.csv").exists()
    assert (output_root / "processed_roi8_cnn_only_prediction_distribution.csv").exists()
    assert (output_root / "processed_roi8_cnn_only_overall_confusion_matrix.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_reports.py::test_report_notebook_rebuilds_summary_outputs_from_saved_predictions -v`

Expected: FAIL with `FileNotFoundError` for `05_regenerate_metrics_and_reports.ipynb`

- [ ] **Step 3: Write minimal implementation**

```python
# 00_shared_setup.ipynb - add code cells
def load_prediction_csvs_for_metrics(seed_metrics_path: Path) -> pd.DataFrame:
    seed_metrics_df = pd.read_csv(seed_metrics_path)
    frames = []
    for prediction_path in seed_metrics_df["predictions_path"].astype(str):
        frames.append(pd.read_csv(prediction_path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def regenerate_official_reports(seed_metrics_path: Path, output_root: Path) -> dict[str, Path]:
    output_root = ensure_dir(output_root)
    seed_metrics_df = pd.read_csv(seed_metrics_path)
    prediction_df = load_prediction_csvs_for_metrics(seed_metrics_path)

    fold_summary_path = output_root / "processed_roi8_cnn_only_fold_summary.csv"
    prediction_distribution_path = output_root / "processed_roi8_cnn_only_prediction_distribution.csv"
    confusion_csv_path = output_root / "processed_roi8_cnn_only_overall_confusion_matrix.csv"
    confusion_png_path = output_root / "processed_roi8_cnn_only_overall_confusion_matrix.png"

    seed_metrics_df.groupby("fold_name", as_index=False)[["accuracy", "macro_precision", "macro_recall", "macro_f1"]].mean().to_csv(fold_summary_path, index=False)
    prediction_df["predicted_label"].value_counts().rename_axis("predicted_label").reset_index(name="count").to_csv(prediction_distribution_path, index=False)

    label_to_index = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    y_true = prediction_df["true_label"].map(label_to_index).astype(int).to_numpy()
    y_pred = prediction_df["predicted_label"].map(label_to_index).astype(int).to_numpy()
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(confusion_csv_path)
    save_confusion_matrix_png(matrix, confusion_png_path, "Overall Confusion Matrix")

    return {
        "fold_summary_path": fold_summary_path,
        "prediction_distribution_path": prediction_distribution_path,
        "confusion_csv_path": confusion_csv_path,
        "confusion_png_path": confusion_png_path,
    }
```

```python
# 05_regenerate_metrics_and_reports.ipynb - code cells
%run ./00_shared_setup.ipynb

TRAINING_OUTPUT_ROOT = Path(
    str(
        override(
            "TRAINING_OUTPUT_ROOT",
            TRAINING_OUTPUTS_ROOT / "mobilenetv3small_8fold_processed_roi_cnn_only",
        )
    )
)
SEED_METRICS_PATH = TRAINING_OUTPUT_ROOT / "processed_roi8_cnn_only_seed_metrics.csv"

report_paths = regenerate_official_reports(SEED_METRICS_PATH, TRAINING_OUTPUT_ROOT)
display(pd.DataFrame([{"name": key, "path": str(value)} for key, value in report_paths.items()]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_reports.py -v`

Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add 00_shared_setup.ipynb 05_regenerate_metrics_and_reports.ipynb tests/test_notebook_reports.py
git commit -m "feat: add notebook metrics regeneration flow"
```

### Task 6: Build The Final Deployment Training Notebook

**Files:**
- Modify: `00_shared_setup.ipynb`
- Create: `06_train_final_deployment_model.ipynb`
- Test: `tests/test_notebook_final_deployment.py`

**Interfaces:**
- Consumes:
  - `build_mobilenetv3small_cnn_only_model(...) -> tf.keras.Model`
  - `compute_class_weights_from_labels(labels: np.ndarray) -> dict[int, float]`
  - `ImageOnlySequence`
  - `classification_metrics(...) -> dict[str, object]`
- Produces:
  - `split_final_deployment_df(processed_df: pd.DataFrame, random_state: int, val_size: float) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `train_final_deployment_model(processed_manifest_path: Path, output_root: Path) -> dict[str, object]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_final_deployment.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from tests.notebook_test_utils import execute_notebook


def test_final_deployment_notebook_writes_model_metadata_and_validation_predictions(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed_hsv_lab_threshold_roi_224"
    rows = []
    for sample_idx in range(1, 9):
        for label_idx, label in enumerate(["fresh", "not fresh", "spoiled"]):
            for image_idx in range(2):
                image_name = f"sample_{sample_idx}_{label}_{image_idx}.jpg".replace(" ", "_")
                image_path = processed_root / label / image_name
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_uint8 = np.full((224, 224, 3), 60 + sample_idx * 10 + label_idx * 20, dtype=np.uint8)
                Image.fromarray(image_uint8).save(image_path)
                rows.append(
                    {
                        "image_name": image_name,
                        "label": label,
                        "sample_id": f"sample_{sample_idx}",
                        "processed_image_path": str(image_path),
                    }
                )

    processed_manifest_path = tmp_path / "processed_manifest.csv"
    pd.DataFrame(rows).to_csv(processed_manifest_path, index=False)

    output_root = tmp_path / "training_outputs" / "mobilenetv3small_8samples_final_deployment_cnn_only"
    execute_notebook(
        Path("06_train_final_deployment_model.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "PROCESSED_MANIFEST_PATH": str(processed_manifest_path),
            "FINAL_OUTPUT_ROOT": str(output_root),
            "MODEL_WEIGHTS": None,
            "BATCH_SIZE": 4,
            "EPOCHS_HEAD": 1,
            "EPOCHS_FINE": 1,
            "FINAL_VAL_SIZE": 0.25,
            "USE_TRAINING_AUGMENTATION": False,
        },
    )

    assert (output_root / "models" / "meatlens_final_8samples_cnn_only_mobilenetv3small.keras").exists()
    assert (output_root / "models" / "meatlens_final_8samples_cnn_only_mobilenetv3small_metadata.json").exists()
    assert (output_root / "final_validation_predictions.csv").exists()
    assert (output_root / "final_training_history.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_final_deployment.py::test_final_deployment_notebook_writes_model_metadata_and_validation_predictions -v`

Expected: FAIL with `FileNotFoundError` for `06_train_final_deployment_model.ipynb`

- [ ] **Step 3: Write minimal implementation**

```python
# 00_shared_setup.ipynb - add code cells
from sklearn.model_selection import train_test_split


def split_final_deployment_df(
    processed_df: pd.DataFrame,
    random_state: int,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_key = processed_df["label"].astype(str)
    train_index, val_index = train_test_split(
        processed_df.index.to_numpy(),
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_key,
    )
    return (
        processed_df.loc[train_index].copy().reset_index(drop=True),
        processed_df.loc[val_index].copy().reset_index(drop=True),
    )


def train_final_deployment_model(processed_manifest_path: Path, output_root: Path) -> dict[str, object]:
    output_root = ensure_dir(output_root)
    models_root = ensure_dir(output_root / "models")
    processed_df = pd.read_csv(processed_manifest_path)
    train_df, val_df = split_final_deployment_df(
        processed_df,
        random_state=int(override("FINAL_DEPLOYMENT_SEED", 42)),
        val_size=float(override("FINAL_VAL_SIZE", 0.15)),
    )

    label_to_index = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    train_labels = train_df["label"].map(label_to_index).astype(int).to_numpy()
    val_labels = val_df["label"].map(label_to_index).astype(int).to_numpy()

    train_sequence = ImageOnlySequence(train_df["processed_image_path"].astype(str).tolist(), train_labels, int(override("BATCH_SIZE", BATCH_SIZE)), True, bool(override("USE_TRAINING_AUGMENTATION", True)))
    val_sequence = ImageOnlySequence(val_df["processed_image_path"].astype(str).tolist(), val_labels, int(override("BATCH_SIZE", BATCH_SIZE)), False, False)

    model = build_mobilenetv3small_cnn_only_model(weights=override("MODEL_WEIGHTS", "imagenet"))
    class_weights = compute_class_weights_from_labels(train_labels)
    callbacks_list = [
        ValMacroF1Callback(val_sequence, val_labels),
        callbacks.EarlyStopping(monitor="val_f1_macro", mode="max", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_f1_macro", mode="max", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]

    model.compile(optimizer=tf.keras.optimizers.Adam(float(override("HEAD_LR", HEAD_LR))), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history_head = model.fit(train_sequence, validation_data=val_sequence, epochs=int(override("EPOCHS_HEAD", EPOCHS_HEAD)), class_weight=class_weights, callbacks=callbacks_list, verbose=2)

    unfreeze_top_backbone_fraction(model, float(override("FINE_TUNE_FRACTION", FINE_TUNE_FRACTION)))
    model.compile(optimizer=tf.keras.optimizers.Adam(float(override("FINE_TUNE_LR", FINE_TUNE_LR))), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history_fine = model.fit(train_sequence, validation_data=val_sequence, epochs=int(override("EPOCHS_FINE", EPOCHS_FINE)), class_weight=class_weights, callbacks=callbacks_list, verbose=2)

    model_path = models_root / "meatlens_final_8samples_cnn_only_mobilenetv3small.keras"
    metadata_path = models_root / "meatlens_final_8samples_cnn_only_mobilenetv3small_metadata.json"
    predictions_path = output_root / "final_validation_predictions.csv"
    history_path = output_root / "final_training_history.csv"

    model.save(model_path, include_optimizer=False)
    y_prob = model.predict(val_sequence, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    metric_row = classification_metrics(val_labels, y_pred)

    prediction_df = val_df.copy()
    prediction_df["true_label"] = val_df["label"]
    prediction_df["predicted_label"] = [LABEL_ORDER[index] for index in y_pred]
    prediction_df["confidence"] = y_prob.max(axis=1)
    prediction_df.to_csv(predictions_path, index=False)

    pd.DataFrame({"phase": ["head", "fine"], "epochs": [len(history_head.epoch), len(history_fine.epoch)]}).to_csv(history_path, index=False)

    metadata = {
        "model_name": "meatlens_final_8samples_cnn_only_mobilenetv3small",
        "backbone": "MobileNetV3Small",
        "model_input_mode": "cnn_only",
        "image_crop_mode": "preprocessed_hsv_lab_threshold_roi_224",
        "input_shape": list(INPUT_SHAPE),
        "target_size": list(TARGET_SIZE),
        "label_order": LABEL_ORDER,
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "validation_accuracy": float(metric_row["accuracy"]),
        "validation_macro_precision": float(metric_row["macro_precision"]),
        "validation_macro_recall": float(metric_row["macro_recall"]),
        "validation_macro_f1": float(metric_row["macro_f1"]),
        "class_weights": class_weights,
        "model_path": str(model_path),
        "deployment_note": "Official evaluation metrics should be taken from the 8-fold cross-rotation experiment, not this final deployment run.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "predictions_path": predictions_path,
        "history_path": history_path,
    }
```

```python
# 06_train_final_deployment_model.ipynb - code cells
%run ./00_shared_setup.ipynb

PROCESSED_MANIFEST_PATH = Path(str(override("PROCESSED_MANIFEST_PATH", GENERATED_SPLITS_ROOT / "processed_manifest.csv")))
FINAL_OUTPUT_ROOT = Path(
    str(
        override(
            "FINAL_OUTPUT_ROOT",
            TRAINING_OUTPUTS_ROOT / "mobilenetv3small_8samples_final_deployment_cnn_only",
        )
    )
)

bundle = train_final_deployment_model(PROCESSED_MANIFEST_PATH, FINAL_OUTPUT_ROOT)
display(pd.DataFrame([{"name": key, "path": str(value)} for key, value in bundle.items()]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_final_deployment.py -v`

Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add 00_shared_setup.ipynb 06_train_final_deployment_model.ipynb tests/test_notebook_final_deployment.py
git commit -m "feat: add final deployment training notebook"
```

### Task 7: Build The ONNX Export And Smoke-Test Notebooks

**Files:**
- Modify: `00_shared_setup.ipynb`
- Create: `07_export_onnx.ipynb`
- Create: `08_inference_smoke_test.ipynb`
- Test: `tests/test_notebook_onnx_workflow.py`

**Interfaces:**
- Consumes:
  - `train_final_deployment_model(...) -> dict[str, object]`
  - `LABEL_ORDER: list[str]`
  - `INPUT_SHAPE: tuple[int, int, int]`
- Produces:
  - `export_model_to_onnx(model_path: Path, onnx_path: Path, opset: int = 13) -> Path`
  - `build_onnx_metadata(model_path: Path, base_metadata: dict[str, object]) -> dict[str, object]`
  - `smoke_test_onnx(onnx_path: Path, batch: np.ndarray) -> dict[str, object]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_onnx_workflow.py
from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from tests.notebook_test_utils import execute_notebook


def test_notebook_onnx_workflow_exports_and_smoke_tests_model(tmp_path: Path) -> None:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(224, 224, 3), name="image_input"),
            tf.keras.layers.Rescaling(scale=1.0 / 255.0),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model_path = tmp_path / "final_model.keras"
    model.save(model_path, include_optimizer=False)

    metadata_path = tmp_path / "final_model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "meatlens_final_8samples_cnn_only_mobilenetv3small",
                "backbone": "MobileNetV3Small",
                "model_input_mode": "cnn_only",
                "image_crop_mode": "preprocessed_hsv_lab_threshold_roi_224",
                "label_order": ["fresh", "not fresh", "spoiled"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    export_root = tmp_path / "training_outputs" / "mobilenetv3small_8samples_final_deployment_cnn_only"
    export_root.mkdir(parents=True, exist_ok=True)

    execute_notebook(
        Path("07_export_onnx.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "FINAL_MODEL_PATH": str(model_path),
            "FINAL_METADATA_PATH": str(metadata_path),
            "ONNX_OUTPUT_PATH": str(export_root / "meatlens_final_8samples_cnn_only_mobilenetv3small.onnx"),
            "ONNX_METADATA_PATH": str(export_root / "meatlens_final_8samples_cnn_only_mobilenetv3small_onnx_metadata.json"),
        },
    )

    execute_notebook(
        Path("08_inference_smoke_test.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "ONNX_OUTPUT_PATH": str(export_root / "meatlens_final_8samples_cnn_only_mobilenetv3small.onnx"),
            "ONNX_SMOKE_SUMMARY_PATH": str(export_root / "onnx_smoke_summary.json"),
        },
    )

    assert (export_root / "meatlens_final_8samples_cnn_only_mobilenetv3small.onnx").exists()
    summary = json.loads((export_root / "onnx_smoke_summary.json").read_text(encoding="utf-8"))
    assert summary["output_shapes"] == [[1, 3]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_onnx_workflow.py::test_notebook_onnx_workflow_exports_and_smoke_tests_model -v`

Expected: FAIL with `FileNotFoundError` for `07_export_onnx.ipynb`

- [ ] **Step 3: Write minimal implementation**

```python
# 00_shared_setup.ipynb - add code cells
import onnx
import onnxruntime as ort
import tf2onnx


def export_model_to_onnx(model_path: Path, onnx_path: Path, opset: int = 13) -> Path:
    ensure_dir(onnx_path.parent)
    model = tf.keras.models.load_model(model_path, compile=False)
    tf2onnx.convert.from_keras(
        model,
        input_signature=(tf.TensorSpec((None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), tf.float32, name="image_input"),),
        opset=opset,
        output_path=str(onnx_path),
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    return onnx_path


def build_onnx_metadata(model_path: Path, base_metadata: dict[str, object]) -> dict[str, object]:
    metadata = dict(base_metadata)
    metadata["onnx_model_path"] = str(model_path)
    metadata["input_shape"] = list(INPUT_SHAPE)
    metadata["target_size"] = list(TARGET_SIZE)
    metadata["label_order"] = LABEL_ORDER
    return metadata


def smoke_test_onnx(onnx_path: Path, batch: np.ndarray) -> dict[str, object]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch.astype(np.float32)})
    return {
        "input_name": input_name,
        "output_shapes": [list(output.shape) for output in outputs],
    }
```

```python
# 07_export_onnx.ipynb - code cells
%run ./00_shared_setup.ipynb

FINAL_MODEL_PATH = Path(str(override("FINAL_MODEL_PATH", TRAINING_OUTPUTS_ROOT / "mobilenetv3small_8samples_final_deployment_cnn_only" / "models" / "meatlens_final_8samples_cnn_only_mobilenetv3small.keras")))
FINAL_METADATA_PATH = Path(str(override("FINAL_METADATA_PATH", TRAINING_OUTPUTS_ROOT / "mobilenetv3small_8samples_final_deployment_cnn_only" / "models" / "meatlens_final_8samples_cnn_only_mobilenetv3small_metadata.json")))
ONNX_OUTPUT_PATH = Path(str(override("ONNX_OUTPUT_PATH", FINAL_MODEL_PATH.with_suffix(".onnx"))))
ONNX_METADATA_PATH = Path(str(override("ONNX_METADATA_PATH", FINAL_METADATA_PATH.with_name("meatlens_final_8samples_cnn_only_mobilenetv3small_onnx_metadata.json"))))

base_metadata = json.loads(FINAL_METADATA_PATH.read_text(encoding="utf-8"))
export_model_to_onnx(FINAL_MODEL_PATH, ONNX_OUTPUT_PATH)
onnx_metadata = build_onnx_metadata(ONNX_OUTPUT_PATH, base_metadata)
ONNX_METADATA_PATH.write_text(json.dumps(onnx_metadata, indent=2), encoding="utf-8")
print(ONNX_OUTPUT_PATH)
print(ONNX_METADATA_PATH)
```

```python
# 08_inference_smoke_test.ipynb - code cells
%run ./00_shared_setup.ipynb

ONNX_OUTPUT_PATH = Path(str(override("ONNX_OUTPUT_PATH", TRAINING_OUTPUTS_ROOT / "mobilenetv3small_8samples_final_deployment_cnn_only" / "models" / "meatlens_final_8samples_cnn_only_mobilenetv3small.onnx")))
ONNX_SMOKE_SUMMARY_PATH = Path(str(override("ONNX_SMOKE_SUMMARY_PATH", TRAINING_OUTPUTS_ROOT / "mobilenetv3small_8samples_final_deployment_cnn_only" / "onnx_smoke_summary.json")))

summary = smoke_test_onnx(ONNX_OUTPUT_PATH, np.zeros((1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), dtype=np.float32))
ensure_dir(ONNX_SMOKE_SUMMARY_PATH.parent)
ONNX_SMOKE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(ONNX_SMOKE_SUMMARY_PATH)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_onnx_workflow.py -v`

Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add 00_shared_setup.ipynb 07_export_onnx.ipynb 08_inference_smoke_test.ipynb tests/test_notebook_onnx_workflow.py
git commit -m "feat: add notebook onnx export workflow"
```

### Task 8: Update The Notebook-First Documentation

**Files:**
- Modify: `docs/training-pipeline-usage.md`
- Test: `tests/test_notebook_docs.py`

**Interfaces:**
- Consumes:
  - notebook filenames from Tasks 1-7
- Produces:
  - notebook-first run instructions in `docs/training-pipeline-usage.md`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_docs.py
from pathlib import Path


def test_training_usage_doc_lists_notebook_run_order() -> None:
    text = Path("docs/training-pipeline-usage.md").read_text(encoding="utf-8")
    assert "00_shared_setup.ipynb" in text
    assert "04_train_8fold_mobilenetv3small.ipynb" in text
    assert "07_export_onnx.ipynb" in text
    assert "Official evaluation metrics come from the 8-fold cross-rotation workflow." in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_docs.py::test_training_usage_doc_lists_notebook_run_order -v`

Expected: FAIL because the current doc does not list the notebook suite

- [ ] **Step 3: Write minimal implementation**

```markdown
# docs/training-pipeline-usage.md
# MeatLens Notebook-First Training Workflow

## Environment

```bash
conda env create -f environment.yml
conda run -n meatlens-pork-training python -m ipykernel install --user --name meatlens-pork-training
```

## Notebook Order

Run these notebooks in order:

1. `00_shared_setup.ipynb`
2. `01_manifest_and_dataset_audit.ipynb`
3. `02_roi_preprocessing.ipynb`
4. `03_build_cross_rotation_splits.ipynb`
5. `04_train_8fold_mobilenetv3small.ipynb`
6. `05_regenerate_metrics_and_reports.ipynb`
7. `06_train_final_deployment_model.ipynb`
8. `07_export_onnx.ipynb`
9. `08_inference_smoke_test.ipynb`

## Official Evaluation

Official evaluation metrics come from the 8-fold cross-rotation workflow.

Use the final deployment and ONNX notebooks only after the official 8-fold workflow is complete.

## Required Inputs

- manifest: `.csv`, `.xlsx`, or `.xls`
- required manifest columns: `image_name`, `label`
- official 8-fold workflow also requires stable `sample_id` grouping
- raw image root folder

## Primary Outputs

- processed ROI dataset
- 8-fold split CSVs
- official metrics and confusion matrices
- final deployment `.keras` model
- ONNX model
- ONNX metadata JSON
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n meatlens-pork-training python -m pytest tests/test_notebook_docs.py -v`

Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add docs/training-pipeline-usage.md tests/test_notebook_docs.py
git commit -m "docs: add notebook-first training usage guide"
```

## Self-Review

### Spec coverage

- Notebook-first multi-notebook codebase architecture: Tasks 1-8
- Shared setup notebook loaded with `%run`: Task 1 and all downstream notebook tasks
- Manifest-driven ingestion: Task 2
- ROI preprocessing contract: Task 2
- Official 8-fold workflow: Tasks 3-5
- MobileNetV3Small CNN-only training defaults: Task 4
- Separate final deployment model: Task 6
- Separate ONNX export and ONNX Runtime smoke test: Task 7
- Required accuracy, precision, recall, F1, predictions, and confusion matrix outputs: Tasks 4-7
- Notebook-first documentation: Task 8

### Placeholder scan

- No `TODO`, `TBD`, or `implement later` markers remain.
- Each task names exact files.
- Each task includes concrete failing tests, commands, and implementation code.

### Type consistency

- `override(...)`, `ensure_dir(...)`, and shared constants originate in Task 1 and are reused consistently later.
- Dataset-flow helpers are introduced in Task 2 and consumed by Task 3.
- Split outputs from Task 3 are consumed by Task 4.
- Training artifacts from Tasks 4 and 6 are consumed by Tasks 5 and 7.
- ONNX helpers introduced in Task 7 are used consistently between export and smoke-test notebooks.
