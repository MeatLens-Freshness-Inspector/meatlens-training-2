from __future__ import annotations

from pathlib import Path


def test_pytest_collection_is_scoped_to_canonical_tests() -> None:
    config_path = Path("pytest.ini")

    assert config_path.exists()

    content = config_path.read_text(encoding="utf-8")

    assert "[pytest]" in content
    assert "testpaths = tests" in content
    assert "norecursedirs = for-deletion" in content


def test_test_suite_uses_unit_and_integration_roots() -> None:
    tests_root = Path("tests")

    assert (tests_root / "unit").is_dir()
    assert (tests_root / "integration").is_dir()
    assert (tests_root / "support").is_dir()

    flat_test_files = sorted(path.name for path in tests_root.glob("test_*.py"))
    assert flat_test_files == []
