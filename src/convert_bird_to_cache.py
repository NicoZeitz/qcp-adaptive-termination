"""
Convert BIRD dataset split folders into cache-format JSON files.

For each q<id> folder found inside data/bird/dataset and data/bird/dataset_vertical,
this script creates a JSON file named q<id>.json in the root of the respective
dataset folder (e.g. data/bird/dataset/q53.json).

The produced files follow the same layout used in data/imdb/queries:
    {
        "extra": {
            "question_id": 53,
            "db_id": "california_schools",
            "SQL": "SELECT ...",
            "difficulty": "simple",
            "answer": [[6], [5], ...]   <- result of running SQL on dev DB
        },
        "value": {
            "__type__": "CollaborativeQueryAnswering",
            "query": "<question> <evidence>",
            "data_agents": [
                {"type": "sql", "name": "A0", "path": "..."},
                ...
            ]
        }
    }

Agent paths are written relative to the src/ directory (where main.py is run from),
matching the convention used in data/imdb/queries files.

Usage:
    python convert_bird_to_cache.py [--dataset-root DATASET_ROOT]
                                    [--dev-json DEV_JSON]
                                    [--dev-db-root DEV_DB_ROOT]
                                    [--src-dir SRC_DIR]

All arguments are optional and default to the standard project layout.
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dev_index(dev_json: Path) -> dict[int, dict]:
    """Return a mapping question_id -> dev entry."""
    with dev_json.open(encoding="utf-8") as f:
        entries = json.load(f)
    return {e["question_id"]: e for e in entries}


def build_query(question: str, evidence: str) -> str:
    """Combine question and evidence into a single query string."""
    q = question.strip()
    e = evidence.strip() if evidence else ""

    if not e:
        return q

    # Keep punctuation readable when appending guidance text.
    if q.endswith((".", "?", "!", ":", ";")):
        return f"{q} Hint: {e}"
    return f"{q}. Hint: {e}"


def run_sql(db_path: Path, sql: str) -> tuple[list, str | None]:
    """
    Execute *sql* on the SQLite database at *db_path*.
    Returns (rows, error_message). rows is a list of lists; error is None on success.
    """
    if not db_path.exists():
        return [], f"Database not found: {db_path}"
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(sql)
        rows = [list(row) for row in cur.fetchall()]
        conn.close()
        return rows, None
    except sqlite3.Error as exc:
        return [], str(exc)


def build_data_agents(q_folder: Path, src_dir: Path) -> list[dict]:
    """
    Discover agent*.sqlite files in *q_folder* and return a data_agents list.
    Paths are expressed relative to *src_dir* (where main.py is executed from).
    """
    agent_files = sorted(q_folder.glob("agent*.sqlite"))
    agents = []
    for af in agent_files:
        # e.g. agent0.sqlite → name A0
        idx = af.stem.replace("agent", "")
        name = f"A{idx}"
        rel_path = os.path.relpath(af.resolve(), src_dir.resolve())
        agents.append({"type": "sql", "name": name, "path": rel_path})
    return agents


def process_dataset(
    dataset_dir: Path,
    dev_index: dict[int, dict],
    dev_db_root: Path,
    src_dir: Path,
) -> tuple[int, int]:
    """
    Walk all q<id> folders inside *dataset_dir* and produce q<id>.json files
    in *dataset_dir* itself.  Returns (converted, skipped).
    """
    converted = skipped = 0

    # q folders can be nested one level deep: <dataset_dir>/<db_name>/<q_id>/
    q_folders = sorted(
        p for p in dataset_dir.glob("**/q*") if p.is_dir() and p.name.startswith("q")
    )

    for q_folder in q_folders:
        folder_name = q_folder.name  # e.g. "q53"
        try:
            question_id = int(folder_name[1:])
        except ValueError:
            print(f"  Skipping non-numeric folder: {q_folder}", file=sys.stderr)
            skipped += 1
            continue

        dev_entry = dev_index.get(question_id)
        if dev_entry is None:
            print(f"  Skipping {folder_name}: question_id {question_id} not found in dev.json", file=sys.stderr)
            skipped += 1
            continue

        db_id: str = dev_entry["db_id"]
        sql: str = dev_entry["SQL"]

        # Build query string
        query = build_query(dev_entry["question"], dev_entry.get("evidence", ""))

        # Execute SQL against the real dev database
        dev_db = dev_db_root / db_id / f"{db_id}.sqlite"
        answer_rows, sql_error = run_sql(dev_db, sql)

        # Build data agents list from the split folder
        data_agents = build_data_agents(q_folder, src_dir)
        if not data_agents:
            print(f"  Warning: no agent*.sqlite files found in {q_folder}", file=sys.stderr)

        extra: dict = {
            "question_id": question_id,
            "db_id": db_id,
            "question": dev_entry["question"],
            "evidence": dev_entry.get("evidence", ""),
            "question_with_evidence": query,
            "SQL": sql,
            "difficulty": dev_entry.get("difficulty"),
            "answer": answer_rows,
        }
        if sql_error:
            extra["answer_error"] = sql_error

        cache_obj = {
            "extra": extra,
            "value": {
                "__type__": "CollaborativeQueryAnswering",
                "query": query,
                "data_agents": data_agents,
            },
        }

        out_file = dataset_dir / f"{folder_name}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(cache_obj, f, indent=4, ensure_ascii=False)

        print(f"  Written: {out_file.relative_to(dataset_dir.parent.parent)}")
        converted += 1

    return converted, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # Script now lives in src/, so project root is one level up.
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Convert BIRD split folders into cache-format JSON files."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(project_root / "data" / "bird"),
        help="Root of the BIRD data directory (default: data/bird)",
    )
    parser.add_argument(
        "--dev-json",
        default=str(project_root / "data" / "bird" / "dev" / "dev.json"),
        help="Path to dev.json (default: data/bird/dev/dev.json)",
    )
    parser.add_argument(
        "--dev-db-root",
        default=str(project_root / "data" / "bird" / "dev" / "dev_databases"),
        help="Root folder of the dev SQLite databases (default: data/bird/dev/dev_databases)",
    )
    parser.add_argument(
        "--src-dir",
        default=str(project_root / "src"),
        help="Directory from which main.py is executed; agent paths are relative to this (default: src/)",
    )
    args = parser.parse_args()

    bird_root = Path(args.dataset_root)
    dev_json = Path(args.dev_json)
    dev_db_root = Path(args.dev_db_root)
    src_dir = Path(args.src_dir)

    dataset_dirs = [
        bird_root / "dataset",
        bird_root / "dataset_vertical",
    ]

    if not dev_json.exists():
        sys.exit(f"Error: dev.json not found at {dev_json}")

    print(f"Loading dev.json from {dev_json} ...")
    dev_index = load_dev_index(dev_json)
    print(f"  Loaded {len(dev_index)} questions.\n")

    total_converted = total_skipped = 0
    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir():
            print(f"Dataset directory not found, skipping: {dataset_dir}")
            continue
        print(f"Processing {dataset_dir} ...")
        c, s = process_dataset(dataset_dir, dev_index, dev_db_root, src_dir)
        total_converted += c
        total_skipped += s
        print()

    print(f"Done. {total_converted} file(s) created, {total_skipped} skipped.")


if __name__ == "__main__":
    main()
