"""
Numpy embedding cache — dimension-sharded.

All vectors of the same dimension are stored in two flat .npy files:
  embeddings_keys_<dim>.npy  — (N,)      SHA-256 hex strings (U64 dtype)
  embeddings_vecs_<dim>.npy  — (N, dim)  float32 matrix

Multiple dimensions coexist in the same cache_dir (e.g. 1536 and 3072 from
different embedding models). Load time for 124k x 3072 vectors: ~2 s.
"""

import json
import logging
import threading
import uuid
from pathlib import Path

import numpy as np

# Sentinel filenames used by context.py to detect whether migration has run.
_KEYS_FILENAME = "embeddings_keys.npy"   # legacy single-dim name, kept for detection only
_VECS_FILENAME = "embeddings_vecs.npy"


def _dim_filenames(dim: int) -> tuple[str, str]:
    return f"embeddings_keys_{dim}.npy", f"embeddings_vecs_{dim}.npy"


class NumpyEmbeddingCache:
    """
    In-memory {hash → row_index} index over float32 matrices, one shard per dimension.
    Writes are deferred: call flush() once after a batch of store() calls.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        # dim → (index, vecs, keys_path, vecs_path)
        self._shards: dict[int, tuple[dict[str, int], list[np.ndarray], Path, Path]] = {}
        self._loaded_dims: set[int] = set()
        self._dirty_dims: set[int] = set()
        self._lock = threading.Lock()

    def _ensure_loaded(self, dim: int) -> None:
        if dim in self._loaded_dims:
            return
        with self._lock:
            if dim in self._loaded_dims:
                return
            keys_name, vecs_name = _dim_filenames(dim)
            keys_path = self._cache_dir / keys_name
            vecs_path = self._cache_dir / vecs_name
            if keys_path.exists() and vecs_path.exists():
                keys_arr = np.load(keys_path, allow_pickle=False)
                vecs_arr = np.load(vecs_path, allow_pickle=False)
                index: dict[str, int] = {k: i for i, k in enumerate(keys_arr.tolist())}
                vecs: list[np.ndarray] = list(vecs_arr)
                logging.info(
                    "NumpyEmbeddingCache: loaded %d vectors (dim=%d) from %s",
                    len(index), dim, vecs_path,
                )
            else:
                index = {}
                vecs = []
                logging.info(
                    "NumpyEmbeddingCache: no cache for dim=%d at %s, starting empty",
                    dim, self._cache_dir,
                )
            self._shards[dim] = (index, vecs, keys_path, vecs_path)
            self._loaded_dims.add(dim)

    def get(self, key_hash: str, dim: int) -> list[float] | None:
        self._ensure_loaded(dim)
        index, vecs, _, _ = self._shards[dim]
        idx = index.get(key_hash)
        if idx is None:
            return None
        return vecs[idx].tolist()

    def get_any_dim(self, key_hash: str, known_dims: list[int]) -> list[float] | None:
        """Look up a hash across the given dimensions (checks each in order)."""
        for dim in known_dims:
            result = self.get(key_hash, dim)
            if result is not None:
                return result
        return None

    def store(self, key_hash: str, vector: list[float]) -> None:
        """Add a vector. Marks dirty but does NOT write to disk — call flush() when done."""
        dim = len(vector)
        self._ensure_loaded(dim)
        with self._lock:
            index, vecs, _, _ = self._shards[dim]
            if key_hash not in index:
                index[key_hash] = len(vecs)
                vecs.append(np.array(vector, dtype="float32"))
                self._dirty_dims.add(dim)

    def flush(self) -> None:
        """Persist pending stores atomically. No-op if nothing changed."""
        with self._lock:
            for dim in list(self._dirty_dims):
                self._flush_dim_locked(dim)
            self._dirty_dims.clear()

    def _flush_dim_locked(self, dim: int) -> None:
        index, vecs, keys_path, vecs_path = self._shards[dim]
        keys_path.parent.mkdir(parents=True, exist_ok=True)
        uid = uuid.uuid4().hex
        tmp_keys = keys_path.with_name(f"{keys_path.stem}_{uid}.tmp.npy")
        tmp_vecs = vecs_path.with_name(f"{vecs_path.stem}_{uid}.tmp.npy")
        try:
            np.save(tmp_keys, np.array(list(index.keys()), dtype="U64"))
            np.save(tmp_vecs, np.stack(vecs, axis=0))
            tmp_keys.replace(keys_path)
            tmp_vecs.replace(vecs_path)
        except Exception:
            for tmp in (tmp_keys, tmp_vecs):
                if tmp.exists():
                    tmp.unlink()
            raise


def migrate_json_to_npy(cache_dir: Path, *, delete_originals: bool = False) -> int:
    """
    One-shot migration: read all Embedding_*.json files in cache_dir and write
    them into per-dimension embeddings_keys_<dim>.npy + embeddings_vecs_<dim>.npy.
    Returns the total number of vectors migrated.

    Safe to call when .npy files already exist — existing hashes are skipped.
    """
    json_files = list(cache_dir.glob("Embedding_*.json"))
    if not json_files:
        print("No Embedding_*.json files found.")
        return 0

    # Group by dimension
    by_dim: dict[int, tuple[list[str], list[np.ndarray]]] = {}
    errors = 0
    print(f"Reading {len(json_files)} JSON files...")

    for i, json_file in enumerate(json_files):
        if i % 10000 == 0:
            print(f"  {i}/{len(json_files)}...")
        hash_key = json_file.stem[len("Embedding_"):]
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
            vec = np.array(payload["value"], dtype="float32")
            dim = len(vec)
            if dim not in by_dim:
                by_dim[dim] = ([], [])
            by_dim[dim][0].append(hash_key)
            by_dim[dim][1].append(vec)
        except Exception as exc:
            logging.warning("NumpyEmbeddingCache migration: skipping %s: %s", json_file.name, exc)
            errors += 1

    total_migrated = 0
    for dim, (new_keys, new_vecs) in by_dim.items():
        keys_name, vecs_name = _dim_filenames(dim)
        keys_path = cache_dir / keys_name
        vecs_path = cache_dir / vecs_name

        existing_index: dict[str, int] = {}
        existing_vecs: list[np.ndarray] = []
        if keys_path.exists() and vecs_path.exists():
            keys_arr = np.load(keys_path, allow_pickle=False)
            vecs_arr = np.load(vecs_path, allow_pickle=False)
            existing_vecs = list(vecs_arr)
            existing_index = {k: i for i, k in enumerate(keys_arr.tolist())}

        # skip already-migrated hashes
        filtered_keys = [k for k in new_keys if k not in existing_index]
        filtered_vecs = [v for k, v in zip(new_keys, new_vecs) if k not in existing_index]

        if not filtered_keys:
            print(f"  dim={dim}: all {len(new_keys)} already migrated, skipping")
            continue

        all_keys = list(existing_index.keys()) + filtered_keys
        all_vecs = existing_vecs + filtered_vecs

        uid = uuid.uuid4().hex
        tmp_keys = keys_path.with_name(f"{keys_path.stem}_{uid}.tmp.npy")
        tmp_vecs = vecs_path.with_name(f"{vecs_path.stem}_{uid}.tmp.npy")
        try:
            print(f"  dim={dim}: writing {len(all_keys)} vectors → {vecs_path}")
            np.save(tmp_keys, np.array(all_keys, dtype="U64"))
            np.save(tmp_vecs, np.stack(all_vecs, axis=0))
            tmp_keys.replace(keys_path)
            tmp_vecs.replace(vecs_path)
        except Exception:
            for tmp in (tmp_keys, tmp_vecs):
                if tmp.exists():
                    tmp.unlink()
            raise

        total_migrated += len(filtered_keys)
        print(f"  dim={dim}: migrated {len(filtered_keys)} vectors ({errors} errors)")

    if delete_originals:
        deleted = 0
        for json_file in json_files:
            try:
                json_file.unlink()
                deleted += 1
            except Exception as exc:
                logging.warning("NumpyEmbeddingCache: could not delete %s: %s", json_file.name, exc)
        print(f"Deleted {deleted} old JSON files")

    return total_migrated
