import argparse
import code
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _normalize_qid(qid: str) -> str:
    cleaned = qid.strip().lower()
    if cleaned.startswith("q"):
        return cleaned
    return f"q{cleaned}"


def _latest_final_result(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("*final_result.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No final_result.json found in {run_dir}")
    return candidates[-1]


def _load_answer(question_file: Path) -> Any:
    payload = json.loads(question_file.read_text(encoding="utf-8"))
    return payload.get("extra", {}).get("answer")


def _load_query(question_file: Path) -> str | None:
    payload = json.loads(question_file.read_text(encoding="utf-8"))
    extra = payload.get("extra", {})
    query = extra.get("question_with_evidence") or extra.get("question")
    if query is None:
        return None
    return str(query)


def _load_sql(question_file: Path) -> str | None:
    payload = json.loads(question_file.read_text(encoding="utf-8"))
    extra = payload.get("extra", {})
    sql = extra.get("SQL")
    if sql is None:
        return None
    return str(sql)


def _split_top_level_csv(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_single = False
    in_double = False

    for ch in text:
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            continue
        if in_single or in_double:
            current.append(ch)
            continue

        if ch == "(":
            depth += 1
            current.append(ch)
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            current.append(ch)
            continue

        if ch == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(ch)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _column_name_from_expr(expr: str) -> str:
    alias_match = re.search(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", expr, flags=re.IGNORECASE)
    if alias_match:
        return alias_match.group(1)

    bare_alias_match = re.search(r"\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", expr)
    if bare_alias_match and ")" in expr:
        return bare_alias_match.group(1)

    expr = expr.strip().strip('"').strip("`")
    if "." in expr:
        expr = expr.split(".")[-1]
    return expr.strip('"').strip("`")


def _infer_answer_columns_from_sql(sql: str | None, width: int) -> list[str]:
    if not sql or width <= 0:
        return [f"col_{i + 1}" for i in range(width)]

    select_match = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql, flags=re.IGNORECASE | re.DOTALL)
    if not select_match:
        return [f"col_{i + 1}" for i in range(width)]

    select_expr = select_match.group(1).strip()
    select_expr = re.sub(r"^DISTINCT\s+", "", select_expr, flags=re.IGNORECASE)
    columns = [_column_name_from_expr(part) for part in _split_top_level_csv(select_expr)]

    if len(columns) != width:
        return [f"col_{i + 1}" for i in range(width)]
    return columns


def _answer_to_df(answer: Any, sql: str | None) -> pd.DataFrame:
    if not isinstance(answer, list):
        return pd.DataFrame()

    rows: list[list[Any]] = []
    for item in answer:
        if isinstance(item, list):
            rows.append(item)
        else:
            rows.append([item])

    if not rows:
        return pd.DataFrame()

    width = max(len(row) for row in rows)
    padded_rows = [row + [None] * (width - len(row)) for row in rows]
    columns = _infer_answer_columns_from_sql(sql, width)
    return pd.DataFrame(data=padded_rows, columns=columns)


def _load_result_df(final_result_file: Path) -> pd.DataFrame:
    payload = json.loads(final_result_file.read_text(encoding="utf-8"))
    result = payload.get("result", {})
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected result format in {final_result_file}")

    # The result JSON is serialized in DataFrame split-like shape.
    data = result.get("data", [])
    columns = result.get("columns", [])
    index = result.get("index", None)

    df = pd.DataFrame(data=data, columns=columns)
    if index is not None:
        try:
            df.index = index
        except Exception:
            pass
    return df


def _load_result_raw(final_result_file: Path) -> dict[str, Any]:
    payload = json.loads(final_result_file.read_text(encoding="utf-8"))
    result = payload.get("result", {})
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected result format in {final_result_file}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open interactive REPL with answer + result loaded for a Bird q-id"
    )
    parser.add_argument("qid", help="Question id, e.g. q53 or 53")
    parser.add_argument(
        "--dataset",
        choices=["dataset", "dataset_vertical"],
        default="dataset",
        help="Bird split to use (default: dataset)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--no-repl",
        action="store_true",
        help="Only validate loading and print summary; do not start interactive shell",
    )
    args = parser.parse_args()

    qid = _normalize_qid(args.qid)
    repo_root = args.repo_root.resolve()

    question_file = repo_root / "data" / "bird" / args.dataset / f"{qid}.json"
    run_dir = repo_root / "data" / "runs" / "bird" / args.dataset / qid

    if not question_file.exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    final_result_file = _latest_final_result(run_dir)
    answer = _load_answer(question_file)
    query = _load_query(question_file)
    sql = _load_sql(question_file)
    answer_df = _answer_to_df(answer, sql)
    result_raw = _load_result_raw(final_result_file)
    result = _load_result_df(final_result_file)

    if args.no_repl:
        print(f"qid={qid}")
        print(f"question_file={question_file}")
        print(f"final_result_file={final_result_file}")
        print(f"query={query}")
        print(f"sql={sql}")
        print(f"answer_rows={len(answer) if isinstance(answer, list) else 'n/a'}")
        print(f"answer_df_shape={answer_df.shape}")
        print(f"answer_df_columns={list(answer_df.columns)}")
        print(f"result_shape={result.shape}")
        print(f"result_columns={list(result.columns)}")
        return 0

    banner = (
        f"Loaded qid={qid}\n"
        f"- answer: python object from extra.answer\n"
        f"- result: pandas DataFrame from final_result.json\n"
        f"- question_file: {question_file}\n"
        f"- final_result_file: {final_result_file}\n\n"
        "Variables available:\n"
        "  query, sql, answer, answer_df, result, result_raw, qid, question_file, final_result_file, pd\n"
    )

    local_vars = {
        "query": query,
        "sql": sql,
        "answer": answer,
        "answer_df": answer_df,
        "result": result,
        "result_raw": result_raw,
        "qid": qid,
        "question_file": question_file,
        "final_result_file": final_result_file,
        "pd": pd,
    }

    code.interact(banner=banner, local=local_vars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
