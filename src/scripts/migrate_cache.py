"""Migrate the old JSON cache format to the new binary (pickle) format.

Old format: JSON ``{"key", "value"[, "extra"]}`` where pydantic values carry a ``"__type__"`` tag and
embeddings are ``list[float]``. New format: cache files are raw pickled values (no wrapper); input/dataset
files are pickled ``{"value": ..., "extra": ...}`` dicts. Cache filenames are derived by rebuilding the real
key objects via ``_deserialize_old`` and re-hashing with xxh3-128 via orjson, using a ``.pkl`` suffix.

Run against any directory of old ``.json`` files, e.g. the ``.cache`` dir and the dataset input dir::

    python -m scripts.migrate_cache .cache dataset/inputs [--delete]
"""

import argparse
import importlib
import json
import logging
import pickle
import pkgutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ValidationError

from qcp_adaptive_termination.observability.cache import Cache

logger = logging.getLogger(__name__)

_EMBEDDING_PREFIX = "Embedding"


def _build_model_registry() -> dict[str, type[BaseModel]]:
    """Import every package submodule, then collect all `BaseModel` subclasses keyed by class name."""
    package = importlib.import_module("qcp_adaptive_termination")
    for module_info in pkgutil.walk_packages(package.__path__, prefix="qcp_adaptive_termination."):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:
            logger.debug(f"skip import {module_info.name}: {exc}")

    registry: dict[str, type[BaseModel]] = {}
    stack: list[type[BaseModel]] = list(BaseModel.__subclasses__())
    seen: set[type[BaseModel]] = set()
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        registry[cls.__name__] = cls
        stack.extend(cls.__subclasses__())
    return registry


def _deserialize_old(value: Any, registry: dict[str, type[BaseModel]]) -> Any:
    """Inverse of the old JSON `__type__` serialization: rebuild real pydantic objects."""
    if isinstance(value, dict):
        type_name = value.get("__type__")
        if type_name == "dict":
            return {key: _deserialize_old(item, registry) for key, item in value.items() if key != "__type__"}
        if type_name is not None:
            cls = registry.get(type_name)
            if cls is None:
                raise KeyError(f"no model class named {type_name!r} in registry")
            content = {key: item for key, item in value.items() if key != "__type__"}
            return cls.model_validate(content)
        raise ValueError(f"dict without __type__ tag: {list(value)[:5]}")
    if isinstance(value, list):
        return [_deserialize_old(item, registry) for item in value]
    return value


def _prefix_from_stem(stem: str) -> str:
    """Old cache name is ``{prefix}_{sha256hex}``; the hex hash never contains ``_``."""
    if "_" not in stem:
        return ""
    return stem.rsplit("_", 1)[0]


def _migrate_file(path: Path, registry: dict[str, type[BaseModel]], delete: bool) -> str:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "value" not in raw:
        return "skipped"

    is_cache = "key" in raw
    prefix = _prefix_from_stem(path.stem) if is_cache else ""

    if is_cache:
        if prefix == _EMBEDDING_PREFIX:
            value: Any = np.asarray(raw["value"], dtype=np.float32)
            # get_or_compute_batch now uses the item directly as key, not wrapped in {argument_name: item}
            real_key: Any = next(iter(_deserialize_old(raw["key"], registry).values()))
        else:
            value = _deserialize_old(raw["value"], registry)
            real_key = _deserialize_old(raw["key"], registry)
        target = Cache(folder=path.parent).cache_file_for_key(real_key)
        pickle_data: Any = value
    else:
        value = _deserialize_old(raw["value"], registry)
        pickle_data = {"value": value, "extra": raw.get("extra")}
        target = path.with_suffix(".pkl")

    temporary_file = target.with_name(f"{target.stem}_{uuid4().hex}.tmp")
    with temporary_file.open("wb") as f:
        pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_file.replace(target)

    if delete:
        path.unlink()

    print(path.name, flush=True)
    return "migrated"


def _migrate_directory(directory: Path, registry: dict[str, type[BaseModel]], delete: bool) -> dict[str, int]:
    counts = {"migrated": 0, "skipped": 0, "failed": 0}
    for path in sorted(directory.glob("*.json")):
        try:
            outcome = _migrate_file(path, registry, delete)
        except ValidationError as exc:
            logger.warning(f"schema mismatch, skipping {path.name}: {exc.error_count()} error(s)")
            counts["skipped"] += 1
            continue
        except Exception as exc:
            logger.error(f"failed {path.name}: {exc}")
            counts["failed"] += 1
            continue
        counts[outcome] += 1
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate old JSON cache to the new pickle format")
    parser.add_argument(
        "directories",
        nargs="+",
        help="Directories of old .json files (e.g. the .cache dir and the dataset input dir)",
    )
    parser.add_argument("--delete", action="store_true", help="Delete each old .json after successful migration")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    registry = _build_model_registry()
    logger.info(f"model registry: {len(registry)} classes")

    total = {"migrated": 0, "skipped": 0, "failed": 0}
    for directory in args.directories:
        path = Path(directory)
        if not path.is_dir():
            logger.warning(f"not a directory, skipping: {path}")
            continue
        counts = _migrate_directory(path, registry, args.delete)
        logger.info(f"{path}: {counts}")
        for key in total:
            total[key] += counts[key]

    logger.info(f"total: {total}")


if __name__ == "__main__":
    main()
