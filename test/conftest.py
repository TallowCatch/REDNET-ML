from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def module_loader(repo_root: Path) -> Callable[[str], ModuleType]:
    cache: dict[str, ModuleType] = {}

    def _load(relative_path: str) -> ModuleType:
        module_path = repo_root / relative_path
        if not module_path.exists():
            raise FileNotFoundError(f"Missing module path: {module_path}")

        cache_key = hashlib.md5(str(module_path).encode("utf-8")).hexdigest()
        module_name = f"testmod_{cache_key}"
        if module_name in cache:
            return cache[module_name]

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed loading module spec: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cache[module_name] = module
        return module

    return _load
