"""
Evaluate stage-1 recall over a log directory produced by eval-stage1-recall-wiki.sh.

For each query subdirectory, reads:
  - the original input JSON (for ground-truth relevant_data_agents from extra)
  - the most recent stage_1_output_filtered_relevant_agents.json (for final selected agents)
  - the most recent stage_1_output_embedding_pre_filter.json (for embedding-filter intermediate results)

Then prints per-query and aggregate precision / recall / F1, both after embedding filtering
and after the full LLM filtering pipeline.

Usage:
    python3 -m scripts.eval_stage1_recall --log_dir /mnt/labstore/fwolf/.logs/wikidata-federated-overlap-dataset/eval_stage1_recall
    python3 -m scripts.eval_stage1_recall --log_dir ... --input_dir /mnt/labstore/nzeitz/dataset/wikidata-federated-overlap-dataset/input
"""

import argparse
import json
import sys
from pathlib import Path


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
    if candidates:
        return candidates[-1]

    return None


def compute_metrics(relevant: list[str], selected: list[str]) -> tuple[float, float, float]:
    tp = len(set(relevant) & set(selected))
    precision = tp / len(selected) if selected else 0.0
    recall = tp / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


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
    if stage1 is None:
        print(f"  [ERROR] no stage_1 output found for {slug}")
        return {"file": slug, "status": "error", "relevant": relevant}

    embedding_filter = find_latest_json(query_log_dir, "stage_1_output_embedding_pre_filter.json")
    final_selected = [a["name"] for a in stage1.get("relevant_data_agents", [])]

    if not relevant:
        print(f"  [SKIP]  {slug}  (no relevant_data_agents in extra)")
        return {"file": slug, "status": "skip", "relevant": [], "final_selected": final_selected}

    # --- embedding filter metrics (optional) ---
    emb_data: dict | None = None
    if embedding_filter:
        emb_selected: list[str] = embedding_filter.get("embedding_filtered_agents", [])
        emb_total: int = embedding_filter.get("total_agents", 0)
        emb_missed = sorted(set(relevant) - set(emb_selected))
        emb_prec, emb_rec, emb_f1 = compute_metrics(relevant, emb_selected)

        l2_scores: dict[str, float] = embedding_filter.get("l2_scores", {})
        sorted_by_dist = sorted(l2_scores.items(), key=lambda x: x[1])
        percentile_metrics: list[dict] = []
        for pct in range(10, 110, 10):
            k = max(1, round(len(sorted_by_dist) * pct / 100))
            top_k = [name for name, _ in sorted_by_dist[:k]]
            prec, rec, f1 = compute_metrics(relevant, top_k)
            percentile_metrics.append({"pct": pct, "k": k, "precision": prec, "recall": rec, "f1": f1})

        emb_data = {
            "selected": emb_selected,
            "missed": emb_missed,
            "total": emb_total,
            "precision": emb_prec,
            "recall": emb_rec,
            "f1": emb_f1,
            "percentile_metrics": percentile_metrics,
        }

    # --- final metrics ---
    missed = sorted(set(relevant) - set(final_selected))
    extra_selected = sorted(set(final_selected) - set(relevant))
    final_prec, final_rec, final_f1 = compute_metrics(relevant, final_selected)

    status = "ok" if not missed else "miss"
    icon = "✓" if not missed else "✗"

    print(f"  {icon} {slug}")
    print(f"       relevant={relevant}  ({len(relevant)} of {emb_data['total'] if emb_data else '?'} agents)")
    if emb_data:
        print(
            f"       [embedding]  {emb_data['total']} → {len(emb_data['selected'])} agents"
            f"  precision={emb_data['precision']:.3f}"
            f"  recall={emb_data['recall']:.3f}"
            f"  f1={emb_data['f1']:.3f}"
            f"  missed={emb_data['missed']}"
        )
    print(f"       [final]      selected={final_selected}")
    print(f"       missed={missed}  extra={extra_selected}")
    print(f"       precision={final_prec:.3f}  recall={final_rec:.3f}  f1={final_f1:.3f}")

    if emb_data:
        emb_ok = 1 if emb_data["recall"] == 1.0 else 0
        final_ok = 1 if not missed else 0
        print()
        print("  " + "-" * 56)
        print(f"  Perfect recall (final): {final_ok}/1  (embedding): {emb_ok}/1")
        print()
        print("  -- After embedding filter --")
        print(f"  Macro precision : {emb_data['precision']:.3f}")
        print(f"  Macro recall    : {emb_data['recall']:.3f}")
        print(f"  Macro F1        : {emb_data['f1']:.3f}")
        print()
        print("  -- Embedding top-K% breakdown --")
        print(f"  {'%':>4}  {'k':>6}  {'precision':>9}  {'recall':>6}  {'f1':>6}")
        for pm in emb_data["percentile_metrics"]:
            print(f"  {pm['pct']:>3}%  {pm['k']:>6}  {pm['precision']:>9.3f}  {pm['recall']:>6.3f}  {pm['f1']:>6.3f}")
        print()
        print("  -- After full LLM filter --")
        print(f"  Macro precision : {final_prec:.3f}")
        print(f"  Macro recall    : {final_rec:.3f}")
        print(f"  Macro F1        : {final_f1:.3f}")
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
    }
    if emb_data is not None:
        result.update({
            "embedding_total": emb_data["total"],
            "embedding_selected": emb_data["selected"],
            "embedding_missed": emb_data["missed"],
            "embedding_precision": round(emb_data["precision"], 4),
            "embedding_recall": round(emb_data["recall"], 4),
            "embedding_f1": round(emb_data["f1"], 4),
        })
        for pm in emb_data["percentile_metrics"]:
            pct_key = f"emb_pct{pm['pct']}"
            result[pct_key + "_k"] = pm["k"]
            result[pct_key + "_precision"] = round(pm["precision"], 4)
            result[pct_key + "_recall"] = round(pm["recall"], 4)
            result[pct_key + "_f1"] = round(pm["f1"], 4)
    return result


def macro_avg(results: list[dict], key: str) -> float:
    vals = [r[key] for r in results if key in r]
    return sum(vals) / len(vals) if vals else 0.0


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
        print("  -- After embedding filter --")
        print(f"  Macro precision : {macro_avg(scored, 'embedding_precision'):.3f}")
        print(f"  Macro recall    : {macro_avg(scored, 'embedding_recall'):.3f}")
        print(f"  Macro F1        : {macro_avg(scored, 'embedding_f1'):.3f}")
        print()
        print("  -- Embedding top-K% breakdown --")
        print(f"  {'%':>4}  {'k':>6}  {'precision':>9}  {'recall':>6}  {'f1':>6}")
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

    print()
    print("  -- After full LLM filter --")
    print(f"  Macro precision : {macro_avg(scored, 'precision'):.3f}")
    print(f"  Macro recall    : {macro_avg(scored, 'recall'):.3f}")
    print(f"  Macro F1        : {macro_avg(scored, 'f1'):.3f}")

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
            **(
                {
                    "embedding_macro_precision": round(macro_avg(scored, "embedding_precision"), 4),
                    "embedding_macro_recall": round(macro_avg(scored, "embedding_recall"), 4),
                    "embedding_macro_f1": round(macro_avg(scored, "embedding_f1"), 4),
                }
                if has_embedding else {}
            ),
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

    query_dirs = sorted(
        d for d in log_dir.iterdir()
        if d.is_dir() and any(d.glob("*/stage_1_output_filtered_relevant_agents.json"))
    )
    if not query_dirs:
        print(f"No subdirectories found in {log_dir}")
        sys.exit(1)

    print(f"Evaluating stage-1 recall over {len(query_dirs)} queries in {log_dir}\n")

    results = [process_query(d, input_dir) for d in query_dirs]
    print_summary(results, log_dir)


if __name__ == "__main__":
    main()
