from __future__ import annotations

from pathlib import Path


def test_pytest_collection_is_scoped_to_canonical_tests() -> None:
    config_path = Path("pytest.ini")

    assert config_path.exists()

    content = config_path.read_text(encoding="utf-8")

    assert "[pytest]" in content
    assert "testpaths = tests" in content
    assert "norecursedirs = for-deletion" in content
