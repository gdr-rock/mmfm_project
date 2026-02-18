"""Smoke test for CLI execution with dummy data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_smoke_script_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "outputs"

    cmd = [
        sys.executable,
        "scripts/00_smoke_test.py",
        "--use_dummy_data",
        "--dry_run",
        "--output_dir",
        str(output_dir),
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    report = output_dir / "smoke" / "smoke_report.json"
    assert report.exists(), f"Expected report not found: {report}"
