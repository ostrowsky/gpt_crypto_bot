"""Offline readiness command; never connects to an exchange or starts workers."""

import json
import os
import sys
from pathlib import Path

from . import APPLICATION_ID
from .isolation import AppPaths, assert_clean_pythonpath


def main():
    try:
        assert_clean_pythonpath(os.environ.get("PYTHONPATH"))
        paths = AppPaths(Path(__file__).resolve().parents[2])
        if not (paths.root / "pyproject.toml").is_file():
            raise ValueError("Own editable project root is unavailable")
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps({
        "application_id": APPLICATION_ID, "app_root": str(paths.root),
        "order_send_enabled": False, "financial_core_only": True,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
