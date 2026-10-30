"""
Evaluate stage-1 recall over a log directory produced by run-discovery-evaluation-wikidata.sh.

For each query subdirectory, reads:
  - the original input JSON (for ground-truth relevant_data_agents from extra)
  - stage_1_output_filtered_relevant_agents.json (for final selected agents)

Prints per-query and aggregate precision / recall / F1.

Macro averaging: average of per-query metrics (each query weighted equally).
Micro averaging: pool all TP/FP/FN across queries, then compute metrics (larger queries dominate).

Usage:
    python3 -m scripts.evaluate_discovery --log_dir /mnt/labstore/fwolf/.logs/wikidata-federated-overlap-dataset/evaluate_discovery_wikidata_DeepSeek-V4-Flash-0731
    python3 -m scripts.evaluate_discovery --log_dir ... --input_dir /mnt/labstore/nzeitz/dataset/wikidata-federated-overlap-dataset/input
"""

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO


class _Tee:
    """Write to both a primary stream and a file simultaneously."""

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        self._secondary.write(data)
        return self._primary.write(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def find_latest_json(query_log_dir: Path, filename: str) -> dict | None:
    # main writes directly into output_dir; older runs used a timestamped subdir
    direct = query_log_dir / filename
    candidates = [direct] if direct.exists() else sorted(query_log_dir.glob(f"*/{filename}"))
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


def find_input_file(query_log_dir: Path, input_dir: Path | None) -> Path | None:
    slug = query_log_dir.name
    if input_dir:
        p = input_dir / f"{slug}.json"
        if p.exists():
            return p
    # main copies the input file directly into the output dir
    direct = query_log_dir / f"{slug}.json"
    if direct.exists():
        return direct
    candidates = sorted(query_log_dir.glob(f"*/{slug}.json"))
    return candidates[-1] if candidates else None


def find_llm_stats(query_log_dir: Path) -> dict | None:
    usage = find_latest_json(query_log_dir, "usage.json")
    if usage is None:
        return None
    llm = usage.get("llm", {})
    emb = usage.get("embeddings", {})
    # sum input/output across all models
    llm_input = sum(m.get("input_tokens", 0) for m in llm.get("per_model", {}).values())
    llm_output = sum(m.get("output_tokens", 0) for m in llm.get("per_model", {}).values())
    return {
        "llm_calls": llm.get("runs", 0),
        "input_tokens": llm_input,
        "output_tokens": llm_output,
        "total_tokens": llm_input + llm_output,
        "llm_duration_ms": round(usage.get("time_s", 0) * 1000),
        "embedding_calls": emb.get("calls", 0),
        "embedding_tokens": sum(m.get("input_tokens", 0) for m in emb.get("per_model", {}).values()),
        "embedding_duration_ms": 0,
        "ranking_duration_ms": 0,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(relevant: list[str], selected: list[str]) -> tuple[float, float, float, int, int, int]:
    """Return (precision, recall, f1, tp, fp, fn)."""
    tp = len(set(relevant) & set(selected))
    fp = len(set(selected) - set(relevant))
    fn = len(set(relevant) - set(selected))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def micro_metrics(tp_total: int, fp_total: int, fn_total: int) -> tuple[float, float, float]:
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def macro_avg(results: list[dict], key: str) -> float:
    vals = [r[key] for r in results if key in r]
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Per-query processing
# ---------------------------------------------------------------------------

def process_query(query_log_dir: Path, input_dir: Path | None) -> dict:
    slug = query_log_dir.name

    input_file = find_input_file(query_log_dir, input_dir)
    if input_file is None:
        print(f"  [ERROR] input file not found for {slug}")
        return {"file": slug, "status": "error"}

    with open(input_file) as f:
        input_data = json.load(f)

    relevant: list[str] = input_data.get("extra", {}).get("relevant_data_agents", [])

    stage1 = find_latest_json(query_log_dir, "stage_1_output_filtered_relevant_agents.json")
    llm_stats = find_llm_stats(query_log_dir)

    if stage1 is None:
        print(f"  [ERROR] no stage_1 output found for {slug}")
        return {"file": slug, "status": "error", "relevant": relevant}

    final_selected = [a["name"] for a in stage1.get("relevant_data_agents", [])]

    if not relevant:
        print(f"  [SKIP]  {slug}  (no relevant_data_agents in extra)")
        return {"file": slug, "status": "skip", "relevant": [], "final_selected": final_selected}

    # --- final metrics ---
    missed = sorted(set(relevant) - set(final_selected))
    extra_selected = sorted(set(final_selected) - set(relevant))
    final_prec, final_rec, final_f1, final_tp, final_fp, final_fn = compute_metrics(relevant, final_selected)

    status = "ok" if not missed else "miss"
    icon = "✓" if not missed else "✗"

    total_agents = len(stage1.get("data_agents", []))

    print(f"  {icon} {slug}")
    print(f"       relevant={relevant}  ({len(relevant)} of {total_agents} agents)")
    print(f"       selected={final_selected}")
    print(f"       missed={missed}  extra={extra_selected}")
    print(f"       precision={final_prec:.3f}  recall={final_rec:.3f}  f1={final_f1:.3f}")
    if llm_stats:
        emb_calls = llm_stats.get("embedding_calls", 0)
        emb_tokens = llm_stats.get("embedding_tokens", 0)
        print(
            f"       [llm stats]  calls={llm_stats['llm_calls']}"
            f"  input_tokens={llm_stats['input_tokens']}"
            f"  output_tokens={llm_stats['output_tokens']}"
            f"  total_tokens={llm_stats['total_tokens']}"
            f"  duration={llm_stats['llm_duration_ms']}ms"
        )
        if emb_calls:
            print(
                f"       [emb stats]  embedding_calls={emb_calls}"
                f"  embedding_tokens={emb_tokens}"
            )

    result: dict = {
        "file": slug,
        "status": status,
        "relevant": relevant,
        "final_selected": final_selected,
        "missed": missed,
        "extra_selected": extra_selected,
        "total_agents": total_agents,
        "precision": round(final_prec, 4),
        "recall": round(final_rec, 4),
        "f1": round(final_f1, 4),
        "tp": final_tp,
        "fp": final_fp,
        "fn": final_fn,
    }
    if llm_stats:
        result.update({
            "llm_calls": llm_stats["llm_calls"],
            "llm_input_tokens": llm_stats["input_tokens"],
            "llm_output_tokens": llm_stats["output_tokens"],
            "llm_total_tokens": llm_stats["total_tokens"],
            "llm_duration_ms": llm_stats["llm_duration_ms"],
            "embedding_calls": llm_stats.get("embedding_calls", 0),
            "embedding_tokens": llm_stats.get("embedding_tokens", 0),
        })
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: list[dict], log_dir: Path) -> None:
    total = len(results)
    ok = sum(1 for r in results if r.get("status") == "ok")
    missed_count = sum(1 for r in results if r.get("status") == "miss")
    errors = sum(1 for r in results if r.get("status") == "error")
    skipped = sum(1 for r in results if r.get("status") == "skip")
    evaluated = total - skipped - errors

    scored = [r for r in results if r.get("status") in ("ok", "miss")]

    avg_relevant = sum(len(r["relevant"]) for r in scored) / len(scored) if scored else 0.0
    avg_total = sum(r.get("total_agents", 0) for r in scored) / len(scored) if scored else 0.0

    print()
    print("=" * 60)
    print(f"  Log dir : {log_dir}")
    print(f"  Queries : {total}  (skipped={skipped}, errors={errors})")
    print(f"  Avg relevant agents : {avg_relevant:.1f} / {avg_total:.0f} total")
    print(f"  Perfect recall (final): {ok}/{evaluated}")

    has_llm_stats = any("llm_total_tokens" in r for r in scored)
    if has_llm_stats:
        total_calls = sum(r.get("llm_calls", 0) for r in scored)
        total_input = sum(r.get("llm_input_tokens", 0) for r in scored)
        total_output = sum(r.get("llm_output_tokens", 0) for r in scored)
        total_tokens = sum(r.get("llm_total_tokens", 0) for r in scored)
        total_dur_ms = sum(r.get("llm_duration_ms", 0) for r in scored)
        avg_calls = total_calls / len(scored) if scored else 0
        avg_tokens = total_tokens / len(scored) if scored else 0
        total_emb_calls = sum(r.get("embedding_calls", 0) for r in scored)
        total_emb_tokens = sum(r.get("embedding_tokens", 0) for r in scored)
        print()
        print("  -- LLM usage (Stage 1) --")
        print(f"  Total LLM calls      : {total_calls}  (avg {avg_calls:.1f}/query)")
        print(f"  Total input tokens   : {total_input:,}")
        print(f"  Total output tokens  : {total_output:,}")
        print(f"  Total tokens         : {total_tokens:,}  (avg {avg_tokens:,.0f}/query)")
        print(f"  Total duration       : {total_dur_ms / 1000:.1f}s  (avg {total_dur_ms / len(scored) / 1000:.1f}s/query)")
        if total_emb_calls:
            print(f"  Total emb calls      : {total_emb_calls}  (avg {total_emb_calls / len(scored):.1f}/query)")
            print(f"  Total emb tokens     : {total_emb_tokens:,}  (avg {total_emb_tokens / len(scored):,.0f}/query)")

    print()
    print("  -- Final --")

    macro_p = macro_avg(scored, "precision")
    macro_r = macro_avg(scored, "recall")
    macro_f = macro_avg(scored, "f1")
    micro_tp = sum(r.get("tp", 0) for r in scored)
    micro_fp = sum(r.get("fp", 0) for r in scored)
    micro_fn = sum(r.get("fn", 0) for r in scored)
    micro_p, micro_r, micro_f = micro_metrics(micro_tp, micro_fp, micro_fn)

    print(f"  Macro precision : {macro_p:.3f}  |  Micro precision : {micro_p:.3f}")
    print(f"  Macro recall    : {macro_r:.3f}  |  Micro recall    : {micro_r:.3f}")
    print(f"  Macro F1        : {macro_f:.3f}  |  Micro F1        : {micro_f:.3f}")

    if missed_count > 0:
        print()
        print("  Queries with missed relevant agents:")
        for r in results:
            if r.get("status") == "miss":
                print(f"    {r['file']}  missed={r['missed']}  selected={r['final_selected']}")

    print("=" * 60)

    summary_path = log_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(
        {
            "total": total, "ok": ok, "missed": missed_count, "errors": errors, "skipped": skipped,
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f, 4),
            "micro_precision": round(micro_p, 4),
            "micro_recall": round(micro_r, 4),
            "micro_f1": round(micro_f, 4),
            **(
                {
                    "llm_calls_total": sum(r.get("llm_calls", 0) for r in scored),
                    "llm_input_tokens_total": sum(r.get("llm_input_tokens", 0) for r in scored),
                    "llm_output_tokens_total": sum(r.get("llm_output_tokens", 0) for r in scored),
                    "llm_total_tokens_total": sum(r.get("llm_total_tokens", 0) for r in scored),
                }
                if has_llm_stats else {}
            ),
            "results": results,
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"\n  Full results -> {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stage-1 recall from run logs")
    parser.add_argument("--log_dir", required=True, help="Base log dir, e.g. .logs/<dataset>/<run_name>")
    parser.add_argument("--input_dir", default=None, help="Input dir with original JSON files (optional)")
    parser.add_argument("--output_dir", default=None, help="Directory to save a copy of the printed output as evaluate_discovery.log")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log dir not found: {log_dir}")
        sys.exit(1)

    input_dir = Path(args.input_dir) if args.input_dir else None

    log_file = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f"{log_dir.name}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, log_file)  # type: ignore[assignment]
        print(f"Logging output to {log_path}")

    try:
        query_dirs = sorted(
            d for d in log_dir.iterdir()
            if d.is_dir() and (
                (d / "stage_1_output_filtered_relevant_agents.json").exists()
                or any(d.glob("*/stage_1_output_filtered_relevant_agents.json"))
            )
        )
        if not query_dirs:
            print(f"No subdirectories found in {log_dir}")
            sys.exit(1)

        print(f"Evaluating stage-1 recall over {len(query_dirs)} queries in {log_dir}\n")

        results = [process_query(d, input_dir) for d in query_dirs]
        print_summary(results, log_dir)
    finally:
        if log_file is not None:
            sys.stdout = sys.__stdout__
            log_file.close()


if __name__ == "__main__":
    main()
