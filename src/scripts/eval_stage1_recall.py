"""
Evaluate stage-1 recall over a log directory produced by eval-stage1-recall-wiki.sh.

For each query subdirectory, reads:
  - the original input JSON (for ground-truth relevant_data_agents from extra)
  - the most recent stage_1_output_filtered_relevant_agents.json (for selected agents)

Then prints per-query and aggregate precision / recall / F1.

Usage:
    python3 -m scripts.eval_stage1_recall --log_dir /mnt/labstore/fwolf/.logs/wikidata-federated-overlap-dataset/eval_stage1_recall
    python3 -m scripts.eval_stage1_recall --log_dir ... --input_dir /mnt/labstore/nzeitz/dataset/wikidata-federated-overlap-dataset/input
"""

import argparse
import json
import sys
from pathlib import Path


def find_stage1_output(query_log_dir: Path) -> dict | None:
    """Find the most recent stage_1_output file under any timestamped subdir."""
    candidates = sorted(query_log_dir.glob("*/stage_1_output_filtered_relevant_agents.json"))
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


def find_input_file(query_log_dir: Path, input_dir: Path | None) -> Path | None:
    """Locate the original input JSON for a query."""
    slug = query_log_dir.name

    if input_dir:
        p = input_dir / f"{slug}.json"
        if p.exists():
            return p

    # fallback: the run script copies the input file into the output dir
    candidates = sorted(query_log_dir.glob(f"*/{slug}.json"))
    if candidates:
        return candidates[-1]

    return None


def process_query(query_log_dir: Path, input_dir: Path | None) -> dict:
    slug = query_log_dir.name

    input_file = find_input_file(query_log_dir, input_dir)
    if input_file is None:
        print(f"  [ERROR] input file not found for {slug}")
        return {"file": slug, "status": "error", "relevant": [], "selected": None, "missed": [], "extra_selected": []}

    with open(input_file) as f:
        input_data = json.load(f)

    extra = input_data.get("extra", {})
    relevant = extra.get("relevant_data_agents", [])

    stage1 = find_stage1_output(query_log_dir)
    if stage1 is None:
        print(f"  [ERROR] no stage_1 output found for {slug}")
        return {"file": slug, "status": "error", "relevant": relevant, "selected": None, "missed": [], "extra_selected": []}

    selected = [a["name"] for a in stage1.get("relevant_data_agents", [])]

    if not relevant:
        print(f"  [SKIP]  {slug}  (no relevant_data_agents in extra)")
        return {"file": slug, "status": "skip", "relevant": [], "selected": selected, "missed": [], "extra_selected": []}

    missed = sorted(set(relevant) - set(selected))
    extra_selected = sorted(set(selected) - set(relevant))

    tp = len(set(relevant) & set(selected))
    precision = tp / len(selected) if selected else 0.0
    recall = tp / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    status = "ok" if not missed else "miss"
    icon = "✓" if not missed else "✗"

    print(f"  {icon} {slug}")
    print(f"       relevant={relevant}  selected={selected}")
    print(f"       missed={missed}  extra={extra_selected}")
    print(f"       precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")

    return {
        "file": slug,
        "status": status,
        "relevant": relevant,
        "selected": selected,
        "missed": missed,
        "extra_selected": extra_selected,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def print_summary(results: list[dict], log_dir: Path) -> None:
    total = len(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    missed = sum(1 for r in results if r["status"] == "miss")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skip")
    evaluated = total - skipped - errors

    scored = [r for r in results if r["status"] in ("ok", "miss")]
    macro_precision = sum(r["precision"] for r in scored) / len(scored) if scored else 0.0
    macro_recall = sum(r["recall"] for r in scored) / len(scored) if scored else 0.0
    macro_f1 = sum(r["f1"] for r in scored) / len(scored) if scored else 0.0

    print()
    print("=" * 60)
    print(f"  Log dir : {log_dir}")
    print(f"  Queries : {total}  (skipped={skipped}, errors={errors})")
    print(f"  Perfect recall: {ok}/{evaluated}")
    print(f"  Macro precision : {macro_precision:.3f}")
    print(f"  Macro recall    : {macro_recall:.3f}")
    print(f"  Macro F1        : {macro_f1:.3f}")

    if missed > 0:
        print()
        print("  Queries with missed relevant agents:")
        for r in results:
            if r["status"] == "miss":
                print(f"    {r['file']}  missed={r['missed']}  selected={r['selected']}")

    print("=" * 60)

    summary_path = log_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(
        {
            "total": total, "ok": ok, "missed": missed, "errors": errors, "skipped": skipped,
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "results": results,
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"\n  Full results -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stage-1 recall from run logs")
    parser.add_argument("--log_dir", required=True, help="Base log dir, e.g. .logs/<dataset>/<run_name>")
    parser.add_argument("--input_dir", default=None, help="Input dir with original JSON files (optional)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log dir not found: {log_dir}")
        sys.exit(1)

    input_dir = Path(args.input_dir) if args.input_dir else None

    query_dirs = sorted(d for d in log_dir.iterdir() if d.is_dir())
    if not query_dirs:
        print(f"No subdirectories found in {log_dir}")
        sys.exit(1)

    print(f"Evaluating stage-1 recall over {len(query_dirs)} queries in {log_dir}\n")

    results = [process_query(d, input_dir) for d in query_dirs]
    print_summary(results, log_dir)


if __name__ == "__main__":
    main()
