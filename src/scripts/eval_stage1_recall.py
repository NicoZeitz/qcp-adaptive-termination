"""
Evaluate stage-1 recall over a log directory produced by eval-stage1-recall-wiki.sh.

For each query subdirectory, reads:
  - the original input JSON (for ground-truth relevant_data_agents from extra)
  - the most recent stage_1_output_filtered_relevant_agents.json (for final selected agents)
  - the most recent stage_1_output_embedding_pre_filter.json (for embedding-filter intermediate results)

Pipeline: embedding filter → LLM filter (= final).

Prints per-query and aggregate precision / recall / F1 after each stage.

Macro averaging: average of per-query metrics (each query weighted equally).
Micro averaging: pool all TP/FP/FN across queries, then compute metrics (larger queries dominate).

Usage:
    python3 -m scripts.eval_stage1_recall --log_dir /mnt/labstore/fwolf/.logs/wikidata-federated-overlap-dataset/eval_stage1_recall
    python3 -m scripts.eval_stage1_recall --log_dir ... --input_dir /mnt/labstore/nzeitz/dataset/wikidata-federated-overlap-dataset/input
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
    candidates = sorted(query_log_dir.glob(f"*/{filename}"))
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
    candidates = sorted(query_log_dir.glob(f"*/{slug}.json"))
    return candidates[-1] if candidates else None


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
    embedding_filter = find_latest_json(query_log_dir, "stage_1_output_embedding_pre_filter.json")

    if stage1 is None and embedding_filter is None:
        print(f"  [ERROR] no stage_1 output found for {slug}")
        return {"file": slug, "status": "error", "relevant": relevant}

    if stage1 is not None:
        final_selected = [a["name"] for a in stage1.get("relevant_data_agents", [])]
    else:
        final_selected = embedding_filter.get("embedding_filtered_agents", [])  # type: ignore[union-attr]

    if not relevant:
        print(f"  [SKIP]  {slug}  (no relevant_data_agents in extra)")
        return {"file": slug, "status": "skip", "relevant": [], "final_selected": final_selected}

    # --- embedding filter metrics (optional) ---
    emb_data: dict | None = None
    if embedding_filter:
        emb_selected: list[str] = embedding_filter.get("embedding_filtered_agents", [])
        emb_total: int = embedding_filter.get("total_agents", 0)
        emb_missed = sorted(set(relevant) - set(emb_selected))
        emb_prec, emb_rec, emb_f1, emb_tp, emb_fp, emb_fn = compute_metrics(relevant, emb_selected)

        llm_selected: list[str] = embedding_filter.get("after_llm_filter_agents", [])
        llm_prec, llm_rec, llm_f1, llm_tp, llm_fp, llm_fn = (
            compute_metrics(relevant, llm_selected) if llm_selected else (0.0, 0.0, 0.0, 0, 0, 0)
        )

        l2_scores: dict[str, float] = embedding_filter.get("l2_scores", {})
        sorted_by_dist = sorted(l2_scores.items(), key=lambda x: x[1])
        percentile_metrics: list[dict] = []
        for pct in range(10, 110, 10):
            k = max(1, round(len(sorted_by_dist) * pct / 100))
            top_k = [name for name, _ in sorted_by_dist[:k]]
            prec, rec, f1, tp, fp, fn = compute_metrics(relevant, top_k)
            percentile_metrics.append({"pct": pct, "k": k, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn})

        emb_data = {
            "selected": emb_selected,
            "missed": emb_missed,
            "total": emb_total,
            "precision": emb_prec,
            "recall": emb_rec,
            "f1": emb_f1,
            "tp": emb_tp,
            "fp": emb_fp,
            "fn": emb_fn,
            "percentile_metrics": percentile_metrics,
            "llm_selected": llm_selected,
            "llm_precision": llm_prec,
            "llm_recall": llm_rec,
            "llm_f1": llm_f1,
            "llm_tp": llm_tp,
            "llm_fp": llm_fp,
            "llm_fn": llm_fn,
        }

    # --- final metrics ---
    missed = sorted(set(relevant) - set(final_selected))
    extra_selected = sorted(set(final_selected) - set(relevant))
    final_prec, final_rec, final_f1, final_tp, final_fp, final_fn = compute_metrics(relevant, final_selected)

    status = "ok" if not missed else "miss"
    icon = "✓" if not missed else "✗"

    print(f"  {icon} {slug}")
    print(f"       relevant={relevant}  ({len(relevant)} of {emb_data['total'] if emb_data else '?'} agents)")
    if emb_data:
        if emb_data["llm_selected"]:
            llm_missed = sorted(set(relevant) - set(emb_data["llm_selected"]))
            print(
                f"       [llm]        {len(emb_data['selected'])} → {len(emb_data['llm_selected'])} agents"
                f"  precision={emb_data['llm_precision']:.3f}"
                f"  recall={emb_data['llm_recall']:.3f}"
                f"  f1={emb_data['llm_f1']:.3f}"
                f"  missed={llm_missed}"
            )
    print(f"       [final]      selected={final_selected}  (= LLM filter output)")
    print(f"       missed={missed}  extra={extra_selected}")
    print(f"       precision={final_prec:.3f}  recall={final_rec:.3f}  f1={final_f1:.3f}")

    if emb_data:
        emb_ok = 1 if emb_data["recall"] == 1.0 else 0
        final_ok = 1 if not missed else 0
        print()
        print("  " + "-" * 56)
        print(f"  Perfect recall (final): {final_ok}/1  (embedding): {emb_ok}/1")
        print()
        if emb_data["llm_selected"]:
            print("  -- After LLM filter (= final) --")
            print(f"  Precision : {emb_data['llm_precision']:.3f}")
            print(f"  Recall    : {emb_data['llm_recall']:.3f}")
            print(f"  F1        : {emb_data['llm_f1']:.3f}")
            print()
        print("  -- Embedding top-K% breakdown --")
        print(f"  {'%':>4}  {'k':>6}  {'precision':>9}  {'recall':>6}  {'f1':>6}")
        n_total = emb_data["total"]
        ratio = len(relevant) / n_total * 100 if n_total else 0
        threshold_pct = next((p for p in range(10, 110, 10) if ratio <= p), None)
        for pm in emb_data["percentile_metrics"]:
            print(f"  {pm['pct']:>3}%  {pm['k']:>6}  {pm['precision']:>9.3f}  {pm['recall']:>6.3f}  {pm['f1']:>6.3f}")
            if threshold_pct and pm["pct"] == threshold_pct:
                print(f"  {'─' * 48}  {len(relevant)}/{n_total} = {ratio:.1f}% relevant")
        print()
        print("  " + "-" * 56)

    result: dict = {
        "file": slug,
        "status": status,
        "relevant": relevant,
        "final_selected": final_selected,
        "missed": missed,
        "extra_selected": extra_selected,
        "precision": round(final_prec, 4),
        "recall": round(final_rec, 4),
        "f1": round(final_f1, 4),
        "tp": final_tp,
        "fp": final_fp,
        "fn": final_fn,
    }
    if emb_data is not None:
        result.update({
            "embedding_total": emb_data["total"],
            "embedding_selected": emb_data["selected"],
            "embedding_missed": emb_data["missed"],
            "embedding_precision": round(emb_data["precision"], 4),
            "embedding_recall": round(emb_data["recall"], 4),
            "embedding_f1": round(emb_data["f1"], 4),
            "embedding_tp": emb_data["tp"],
            "embedding_fp": emb_data["fp"],
            "embedding_fn": emb_data["fn"],
        })
        if emb_data["llm_selected"]:
            result.update({
                "llm_precision": round(emb_data["llm_precision"], 4),
                "llm_recall": round(emb_data["llm_recall"], 4),
                "llm_f1": round(emb_data["llm_f1"], 4),
                "llm_tp": emb_data["llm_tp"],
                "llm_fp": emb_data["llm_fp"],
                "llm_fn": emb_data["llm_fn"],
            })
        for pm in emb_data["percentile_metrics"]:
            pct_key = f"emb_pct{pm['pct']}"
            result[pct_key + "_k"] = pm["k"]
            result[pct_key + "_precision"] = round(pm["precision"], 4)
            result[pct_key + "_recall"] = round(pm["recall"], 4)
            result[pct_key + "_f1"] = round(pm["f1"], 4)
            result[pct_key + "_tp"] = pm["tp"]
            result[pct_key + "_fp"] = pm["fp"]
            result[pct_key + "_fn"] = pm["fn"]
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_chart_js(results: list[dict], log_dir: Path, chart_output: Path) -> None:
    """Write queries.json and copy l2_chart.html into chart_output for local serving."""
    eligible = [r for r in results if r.get("status") in ("ok", "miss", "skip") and r.get("file")]
    if not eligible:
        print("  [chart] No queries to chart — skipping.")
        return

    html_src = Path(__file__).parent / "cutoff" / "l2_chart.html"
    if not html_src.exists():
        print(f"  [chart] l2_chart.html not found at {html_src} — skipping.")
        return

    queries: list[dict] = []
    skipped = 0
    for r in eligible:
        query_dir = log_dir / r["file"]
        emb_json_paths = sorted(query_dir.glob("*/stage_1_output_embedding_pre_filter.json"))
        if not emb_json_paths:
            skipped += 1
            continue
        with open(emb_json_paths[-1]) as f:
            emb = json.load(f)

        l2_scores: dict[str, float] = emb.get("l2_scores", {})
        best_columns: dict[str, str] = emb.get("best_columns", {})
        if not l2_scores:
            skipped += 1
            continue

        agent_rows: dict[str, tuple[float, str, str]] = {}
        for name, l2 in l2_scores.items():
            table_col = best_columns.get(name, "unknown.unknown")
            parts = table_col.split(".", 1)
            table = parts[0] if len(parts) == 2 else "unknown"
            col   = parts[1] if len(parts) == 2 else table_col
            agent_rows[name] = (l2, f"main.{table}", col)

        sorted_agents = sorted(agent_rows.items(), key=lambda x: x[1][0])
        relevant_names = r.get("relevant", [])

        queries.append({
            "label": r["file"].replace("_", " "),
            "relevant": ",".join(sorted(relevant_names)),
            "raw": "\n".join(
                f"{name},{l2:.4f},{table},{col}"
                for name, (l2, table, col) in sorted_agents
            ),
            "cutoffs": [],
        })

    if not queries:
        print("  [chart] No embedding data found in any query log — skipping.")
        return

    chart_output.mkdir(parents=True, exist_ok=True)

    queries_path = chart_output / "queries.json"
    queries_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")

    out_path = chart_output / "l2_chart.html"
    out_path.write_text(html_src.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"  [chart] Written {len(queries)} queries -> {queries_path}")
    if skipped:
        print(f"  [chart] Skipped {skipped} queries without embedding data")
    print(f"  [chart] Serve with: python3 -m http.server --directory {chart_output}")


def print_summary(results: list[dict], log_dir: Path) -> None:
    total = len(results)
    ok = sum(1 for r in results if r.get("status") == "ok")
    missed_count = sum(1 for r in results if r.get("status") == "miss")
    errors = sum(1 for r in results if r.get("status") == "error")
    skipped = sum(1 for r in results if r.get("status") == "skip")
    evaluated = total - skipped - errors

    scored = [r for r in results if r.get("status") in ("ok", "miss")]
    has_embedding = any("embedding_precision" in r for r in scored)

    avg_relevant = sum(len(r["relevant"]) for r in scored) / len(scored) if scored else 0.0
    avg_total = sum(r.get("embedding_total", r.get("total_agents", 0)) for r in scored) / len(scored) if scored else 0.0

    print()
    print("=" * 60)
    print(f"  Log dir : {log_dir}")
    print(f"  Queries : {total}  (skipped={skipped}, errors={errors})")
    print(f"  Avg relevant agents : {avg_relevant:.1f} / {avg_total:.0f} total")
    print(f"  Perfect recall (final): {ok}/{evaluated}")

    if has_embedding:
        emb_ok = sum(1 for r in scored if r.get("embedding_recall", 0.0) == 1.0)
        print(f"  Perfect recall (embedding): {emb_ok}/{evaluated}")
        print()
        has_llm = any("llm_precision" in r for r in scored)

        if has_llm:
            print("  -- After LLM filter (= final) --")
            macro_p = macro_avg(scored, "llm_precision")
            macro_r = macro_avg(scored, "llm_recall")
            macro_f = macro_avg(scored, "llm_f1")
            micro_tp = sum(r.get("llm_tp", 0) for r in scored)
            micro_fp = sum(r.get("llm_fp", 0) for r in scored)
            micro_fn = sum(r.get("llm_fn", 0) for r in scored)
            micro_p, micro_r, micro_f = micro_metrics(micro_tp, micro_fp, micro_fn)
            print(f"  Macro precision : {macro_p:.3f}  |  Micro precision : {micro_p:.3f}")
            print(f"  Macro recall    : {macro_r:.3f}  |  Micro recall    : {micro_r:.3f}")
            print(f"  Macro F1        : {macro_f:.3f}  |  Micro F1        : {micro_f:.3f}")

        print()
        print("  -- Embedding top-K% breakdown (macro avg) --")
        print(f"  {'%':>4}  {'k':>6}  {'precision':>9}  {'recall':>6}  {'f1':>6}")
        avg_n_relevant = sum(len(r["relevant"]) for r in scored) / len(scored) if scored else 0
        avg_n_total = sum(r.get("embedding_total", 0) for r in scored) / len(scored) if scored else 0
        avg_ratio = avg_n_relevant / avg_n_total * 100 if avg_n_total else 0
        avg_threshold_pct = next((p for p in range(10, 110, 10) if avg_ratio <= p), None)
        for pct in range(10, 110, 10):
            pct_key = f"emb_pct{pct}"
            rows = [r for r in scored if pct_key + "_precision" in r]
            if not rows:
                continue
            avg_k = sum(r[pct_key + "_k"] for r in rows) / len(rows)
            avg_prec = macro_avg(rows, pct_key + "_precision")
            avg_rec = macro_avg(rows, pct_key + "_recall")
            avg_f1 = macro_avg(rows, pct_key + "_f1")
            print(f"  {pct:>3}%  {avg_k:>6.1f}  {avg_prec:>9.3f}  {avg_rec:>6.3f}  {avg_f1:>6.3f}")
            if avg_threshold_pct and pct == avg_threshold_pct:
                print(f"  {'─' * 48}  avg {avg_n_relevant:.1f}/{avg_n_total:.0f} = {avg_ratio:.1f}% relevant")

    print()
    print("  -- Final (= after LLM filter) --")

    # macro
    macro_p = macro_avg(scored, "precision")
    macro_r = macro_avg(scored, "recall")
    macro_f = macro_avg(scored, "f1")
    # micro
    micro_tp = sum(r.get("tp", 0) for r in scored)
    micro_fp = sum(r.get("fp", 0) for r in scored)
    micro_fn = sum(r.get("fn", 0) for r in scored)
    micro_p, micro_r, micro_f = micro_metrics(micro_tp, micro_fp, micro_fn)

    print(f"  Macro precision : {macro_p:.3f}  |  Micro precision : {micro_p:.3f}")
    print(f"  Macro recall    : {macro_r:.3f}  |  Micro recall    : {micro_r:.3f}")
    print(f"  Macro F1        : {macro_f:.3f}  |  Micro F1        : {micro_f:.3f}")

    if missed_count > 0:
        print()
        print("  Queries with missed relevant agents (final):")
        for r in results:
            if r.get("status") == "miss":
                print(f"    {r['file']}  missed={r['missed']}  selected={r['final_selected']}")

    print("=" * 60)

    summary_path = log_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(
        {
            "total": total, "ok": ok, "missed": missed_count, "errors": errors, "skipped": skipped,
            "macro_precision": round(macro_avg(scored, "precision"), 4),
            "macro_recall": round(macro_avg(scored, "recall"), 4),
            "macro_f1": round(macro_avg(scored, "f1"), 4),
            "micro_precision": round(micro_metrics(
                sum(r.get("tp", 0) for r in scored),
                sum(r.get("fp", 0) for r in scored),
                sum(r.get("fn", 0) for r in scored),
            )[0], 4),
            "micro_recall": round(micro_metrics(
                sum(r.get("tp", 0) for r in scored),
                sum(r.get("fp", 0) for r in scored),
                sum(r.get("fn", 0) for r in scored),
            )[1], 4),
            "micro_f1": round(micro_metrics(
                sum(r.get("tp", 0) for r in scored),
                sum(r.get("fp", 0) for r in scored),
                sum(r.get("fn", 0) for r in scored),
            )[2], 4),
            **(
                {
                    "embedding_macro_precision": round(macro_avg(scored, "embedding_precision"), 4),
                    "embedding_macro_recall": round(macro_avg(scored, "embedding_recall"), 4),
                    "embedding_macro_f1": round(macro_avg(scored, "embedding_f1"), 4),
                    "embedding_micro_precision": round(micro_metrics(
                        sum(r.get("embedding_tp", 0) for r in scored),
                        sum(r.get("embedding_fp", 0) for r in scored),
                        sum(r.get("embedding_fn", 0) for r in scored),
                    )[0], 4),
                    "embedding_micro_recall": round(micro_metrics(
                        sum(r.get("embedding_tp", 0) for r in scored),
                        sum(r.get("embedding_fp", 0) for r in scored),
                        sum(r.get("embedding_fn", 0) for r in scored),
                    )[1], 4),
                    "embedding_micro_f1": round(micro_metrics(
                        sum(r.get("embedding_tp", 0) for r in scored),
                        sum(r.get("embedding_fp", 0) for r in scored),
                        sum(r.get("embedding_fn", 0) for r in scored),
                    )[2], 4),
                }
                if has_embedding else {}
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
    parser.add_argument("--output_dir", default=None, help="Directory to save a copy of the printed output as eval_stage1_recall.log")
    parser.add_argument("--chart_output", default=None, help="Path to write a JS snippet (RELEVANT + RAW consts) for l2_chart.html")
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
                any(d.glob("*/stage_1_output_filtered_relevant_agents.json"))
                or any(d.glob("*/stage_1_output_embedding_pre_filter.json"))
            )
        )
        if not query_dirs:
            print(f"No subdirectories found in {log_dir}")
            sys.exit(1)

        print(f"Evaluating stage-1 recall over {len(query_dirs)} queries in {log_dir}\n")

        results = [process_query(d, input_dir) for d in query_dirs]
        print_summary(results, log_dir)
        if args.chart_output:
            write_chart_js(results, log_dir, Path(args.chart_output))
    finally:
        if log_file is not None:
            sys.stdout = sys.__stdout__
            log_file.close()


if __name__ == "__main__":
    main()
