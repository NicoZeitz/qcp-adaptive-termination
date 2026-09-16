"""
Single-file numpy embedding cache.

Replaces the per-file JSON cache for embeddings. All vectors are stored in one
.npy pair (keys + matrix) instead of one JSON file per embedding.

Layout on disk:
  embeddings_keys.npy   — (N,) array of SHA-256 hex strings (U64 dtype)
  embeddings_vecs.npy   — (N, dim) float32 matrix

Load time for 124k × 3072-dim vectors: < 2 seconds (vs ~30 min for 124k JSON files).
"""

import json
import logging
import threading
import uuid
from pathlib import Path

import numpy as np

_KEYS_FILENAME = "embeddings_keys.npy"
_VECS_FILENAME = "embeddings_vecs.npy"


class NumpyEmbeddingCache:
    """
    In-memory {hash → row_index} index over a float32 matrix, backed by two .npy files.
    Writes are deferred: call flush() once after a batch of store() calls.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._keys_path = cache_dir / _KEYS_FILENAME
        self._vecs_path = cache_dir / _VECS_FILENAME
        self._index: dict[str, int] = {}   # hash → row in _vecs
        self._vecs: list[np.ndarray] = []  # list of 1-D float32 arrays
        self._loaded = False
        self._dirty = False
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if self._keys_path.exists() and self._vecs_path.exists():
                keys = np.load(self._keys_path, allow_pickle=False)
                vecs = np.load(self._vecs_path, allow_pickle=False)
                self._vecs = list(vecs)
                self._index = {k: i for i, k in enumerate(keys.tolist())}
                logging.info(
                    "NumpyEmbeddingCache: loaded %d vectors (%s)",
                    len(self._index),
                    self._vecs_path,
                )
            else:
                logging.info(
                    "NumpyEmbeddingCache: no existing cache at %s, starting empty",
                    self._vecs_path,
                )
            self._loaded = True

    def get(self, key_hash: str) -> list[float] | None:
        self._ensure_loaded()
        idx = self._index.get(key_hash)
        if idx is None:
            return None
        return self._vecs[idx].tolist()

    def store(self, key_hash: str, vector: list[float]) -> None:
        """Add a vector. Marks dirty but does NOT write to disk — call flush() when done."""
        self._ensure_loaded()
        with self._lock:
            if key_hash not in self._index:
                self._index[key_hash] = len(self._vecs)
                self._vecs.append(np.array(vector, dtype="float32"))
                self._dirty = True

    def flush(self) -> None:
        """Persist pending stores atomically. No-op if nothing changed."""
        with self._lock:
            if not self._dirty:
                return
            self._flush_locked()
            self._dirty = False

    def _flush_locked(self) -> None:
        """Atomic write of keys + matrix. Must be called with self._lock held."""
        self._keys_path.parent.mkdir(parents=True, exist_ok=True)
        uid = uuid.uuid4().hex
        tmp_keys = self._keys_path.with_name(f"embeddings_keys_{uid}.tmp.npy")
        tmp_vecs = self._vecs_path.with_name(f"embeddings_vecs_{uid}.tmp.npy")
        try:
            keys_arr = np.array(list(self._index.keys()), dtype="U64")
            vecs_arr = np.stack(self._vecs, axis=0)  # (N, dim) float32
            np.save(tmp_keys, keys_arr)
            np.save(tmp_vecs, vecs_arr)
            tmp_keys.replace(self._keys_path)
            tmp_vecs.replace(self._vecs_path)
        except Exception:
            for tmp in (tmp_keys, tmp_vecs):
                if tmp.exists():
                    tmp.unlink()
            raise


def migrate_json_to_npy(cache_dir: Path, *, delete_originals: bool = False) -> int:
    """
    One-shot migration: read all Embedding_*.json files in cache_dir and write
    them into embeddings_keys.npy + embeddings_vecs.npy.
    Returns the number of vectors migrated.

    Safe to call when the .npy files already exist — new entries are appended,
    existing hashes are skipped.
    """
    json_files = list(cache_dir.glob("Embedding_*.json"))
    if not json_files:
        return 0

    keys_path = cache_dir / _KEYS_FILENAME
    vecs_path = cache_dir / _VECS_FILENAME

    existing_index: dict[str, int] = {}
    existing_vecs: list[np.ndarray] = []

    if keys_path.exists() and vecs_path.exists():
        keys_arr = np.load(keys_path, allow_pickle=False)
        vecs_arr = np.load(vecs_path, allow_pickle=False)
        existing_vecs = list(vecs_arr)
        existing_index = {k: i for i, k in enumerate(keys_arr.tolist())}

    new_keys: list[str] = []
    new_vecs: list[np.ndarray] = []
    errors = 0

    for json_file in json_files:
        hash_key = json_file.stem[len("Embedding_"):]
        if hash_key in existing_index:
            continue
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
            vec = np.array(payload["value"], dtype="float32")
            new_keys.append(hash_key)
            new_vecs.append(vec)
        except Exception as exc:
            logging.warning("NumpyEmbeddingCache migration: skipping %s: %s", json_file.name, exc)
            errors += 1

    migrated = len(new_keys)
    if migrated > 0:
        all_keys = list(existing_index.keys()) + new_keys
        all_vecs = existing_vecs + new_vecs

        uid = uuid.uuid4().hex
        tmp_keys = keys_path.with_name(f"embeddings_keys_{uid}.tmp.npy")
        tmp_vecs = vecs_path.with_name(f"embeddings_vecs_{uid}.tmp.npy")
        try:
            np.save(tmp_keys, np.array(all_keys, dtype="U64"))
            np.save(tmp_vecs, np.stack(all_vecs, axis=0))
            tmp_keys.replace(keys_path)
            tmp_vecs.replace(vecs_path)
        except Exception:
            for tmp in (tmp_keys, tmp_vecs):
                if tmp.exists():
                    tmp.unlink()
            raise

        logging.info(
            "NumpyEmbeddingCache: migrated %d vectors → %s (%d errors)",
            migrated,
            vecs_path,
            errors,
        )

    if delete_originals:
        deleted = 0
        for json_file in json_files:
            try:
                json_file.unlink()
                deleted += 1
            except Exception as exc:
                logging.warning("NumpyEmbeddingCache: could not delete %s: %s", json_file.name, exc)
        logging.info("NumpyEmbeddingCache: deleted %d old JSON files", deleted)

    return migrated
