"""Explicit app-local paths and credentials, with no inherited legacy fallback."""

from dataclasses import dataclass, field
from pathlib import Path


class AppPaths:
    def __init__(self, root):
        candidate = Path(root)
        if not candidate.is_absolute():
            raise ValueError("An explicit absolute application root is required")
        self.root = candidate.resolve()

    def _within_root(self, relative):
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("Path must remain inside the application root")
        resolved = (self.root / candidate).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Resolved path escapes the application root")
        return resolved

    def runtime_path(self, relative):
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("Runtime path must be relative without traversal")
        runtime = self._within_root(".runtime")
        resolved = self._within_root(Path(".runtime") / candidate)
        if not resolved.is_relative_to(runtime):
            raise ValueError("Resolved path escapes application runtime")
        return resolved

    @property
    def env_file(self):
        return self._within_root(".env")


def assert_clean_pythonpath(value):
    if value:
        raise ValueError("Injected PYTHONPATH is forbidden for this application")


@dataclass(frozen=True)
class DemoCredentials:
    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)


def load_demo_credentials(paths, explicit=None):
    names = ("BINANCE_DEMO_API_KEY", "BINANCE_DEMO_API_SECRET")
    if explicit is None:
        values = {}
        try:
            content = paths.env_file.read_text(encoding="utf-8-sig")
        except OSError:
            raise ValueError("Own demo credential file is unavailable") from None
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            name = name.strip()
            if name not in names:
                continue
            if not separator or name in values:
                raise ValueError("Malformed or duplicate demo credential field")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[name] = value
    else:
        values = explicit
    pair = [values.get(name) for name in names]
    if any(not isinstance(value, str) or not value.strip() for value in pair):
        raise ValueError("A complete explicit demo credential pair is required")
    return DemoCredentials(*pair)
