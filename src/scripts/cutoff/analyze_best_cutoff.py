import numpy as np
import json

def _load_queries():
    return json.load(open(__file__.replace("analyze_best_cutoff.py", "queries.json")))

def _save_queries(queries):
    json.dump(queries, open(__file__.replace("analyze_best_cutoff.py", "queries.json"), "w"), indent=4, ensure_ascii=False)

queries = _load_queries()

def find_elbow(results: list[dict], key: str = "similarity_score") -> int:
    """Geometric elbow: index of the point with maximum perpendicular distance
    from the line connecting the first and last score (both axes normalized to [0,1])."""
    scores = np.array([float(r[key]) for r in results], dtype=float)
    n = len(scores)
    if n <= 2:
        return n
    x = np.arange(n, dtype=float)
    # Normalize both axes to [0, 1]
    x_range = x[-1] - x[0]
    s_range = scores[-1] - scores[0]
    if x_range == 0 or s_range == 0:
        return n
    xn = (x - x[0]) / x_range
    yn = (scores - scores[0]) / s_range
    # Perpendicular distance from the diagonal line (0,0)→(1,1)
    # Line: y = x  →  distance = |y - x| / sqrt(2)
    distances = np.abs(yn - xn)
    knee = int(np.argmax(distances))
    return knee + 1 # +1 → exclusive cutoff

recalls_largest = []
precisions_largest = []

for query in queries:
    question = query["label"]
    relevant = query["relevant"].split(",")
    rows = [line.split(",") for line in query["raw"].strip().split("\n")]
    results = [{"agent": r[0], "similarity_score": r[1], "database": r[2], "table": r[3].split(".")[0], "column": r[3].split(".")[1]} for r in rows]

    # Find largest gap between two similarity scores
    scores = [float(r["similarity_score"]) for r in results]
    gaps = []
    for i in range(len(scores) - 1):
        if scores[i] == scores[i + 1]:
            continue
        else:
            gaps.append((scores[i + 1] - scores[i], i))

    largest_gap = max(gaps, key=lambda x: x[0]) if gaps else None
    largest_gap_cutoff = largest_gap[1] if largest_gap else None
    print(f"    largest gap: {largest_gap} at index {largest_gap[1]} between scores {scores[largest_gap[1]]} and {scores[largest_gap[1] + 1]}") if largest_gap else print("no gaps found")

    # all agents that are before the cutoff
    agents_before_cutoff = [r["agent"] for r in results[:largest_gap_cutoff + 1]] if largest_gap_cutoff is not None else []
    agents_before_cutoff = set(agents_before_cutoff)

    relevant_agents = set(relevant)
    missing_agents = relevant_agents - agents_before_cutoff

    recall = (len(relevant_agents) - len(missing_agents)) / len(relevant_agents) if len(relevant_agents) > 0 else 0
    precision = (len(relevant_agents) - len(missing_agents)) / len(agents_before_cutoff) if len(agents_before_cutoff) > 0 else 0

    recalls_largest.append(recall)
    precisions_largest.append(precision)

    print(f"{question}:")
    print(f"    relevant agents         : {len(relevant_agents)}")
    print(f"    agents before cutoff    : {len(agents_before_cutoff)}")
    print(f"    missing relevant agents : {len(missing_agents)}")
    print(f"    recall                  : {len(relevant_agents) - len(missing_agents)} / {len(relevant_agents)} = {recall:.2f}")
    print(f"    precision               : {len(relevant_agents) - len(missing_agents)} / {len(agents_before_cutoff)} = {precision:.2f}")

print(f"Average recall across all queries: {sum(recalls_largest) / len(recalls_largest):.2f}")
print(f"Average precision across all queries: {sum(precisions_largest) / len(precisions_largest):.2f}")

print("---------------BORDER-----------------")

recalls_kneedle = []
precisions_kneedle = []

for query in queries:
    question = query["label"]
    relevant = query["relevant"].split(",")
    rows = [line.split(",") for line in query["raw"].strip().split("\n")]
    results = [{"agent": r[0], "similarity_score": r[1], "database": r[2], "table": r[3].split(".")[0], "column": r[3].split(".")[1]} for r in rows]

    # Collapse equal scores before running kneedle (same as "Collapse equal scores" in the chart)
    deduped = []
    for r in results:
        if deduped and float(r["similarity_score"]) == float(deduped[-1]["similarity_score"]):
            deduped[-1]["_agents"].append(r["agent"])
        else:
            deduped.append({**r, "_agents": [r["agent"]]})

    kneedle_cutoff = find_elbow(deduped, key="similarity_score")

    # Expand back: all agents in the deduped groups before the cutoff
    agents_before_cutoff = set(a for g in deduped[:kneedle_cutoff] for a in g["_agents"]) if kneedle_cutoff is not None else set()

    relevant_agents = set(relevant)
    missing_agents = relevant_agents - agents_before_cutoff

    recall = (len(relevant_agents) - len(missing_agents)) / len(relevant_agents) if len(relevant_agents) > 0 else 0
    precision = (len(relevant_agents) - len(missing_agents)) / len(agents_before_cutoff) if len(agents_before_cutoff) > 0 else 0

    recalls_kneedle.append(recall)
    precisions_kneedle.append(precision)

    print(f"{question}:")
    print(f"    relevant agents         : {len(relevant_agents)}")
    print(f"    unique score groups     : {len(deduped)} (from {len(results)} agents)")
    print(f"    agents before cutoff    : {len(agents_before_cutoff)}")
    print(f"    missing relevant agents : {len(missing_agents)}")
    print(f"    recall                  : {len(relevant_agents) - len(missing_agents)} / {len(relevant_agents)} = {recall:.2f}")
    print(f"    precision               : {len(relevant_agents) - len(missing_agents)} / {len(agents_before_cutoff)} = {precision:.2f}")
    print(f"    kneedle cutoff index    : {kneedle_cutoff} (deduped) with score {deduped[kneedle_cutoff - 1]['similarity_score'] if kneedle_cutoff and kneedle_cutoff > 0 else 'N/A'}")

print(f"Average recall across all queries: {sum(recalls_kneedle) / len(recalls_kneedle):.2f}")
print(f"Average precision across all queries: {sum(precisions_kneedle) / len(precisions_kneedle):.2f}")

# ── Write cutoffs back to queries.json ────────────────────────────────────────
print("\nWriting cutoffs to queries.json...")
for query in queries:
    rows = [line.split(",") for line in query["raw"].strip().split("\n")]
    results = [{"agent": r[0], "similarity_score": r[1]} for r in rows]
    scores = [float(r["similarity_score"]) for r in results]

    # Largest gap
    gaps = [(scores[i+1] - scores[i], i) for i in range(len(scores)-1) if scores[i] != scores[i+1]]
    gap_idx = max(gaps, key=lambda x: x[0])[1] if gaps else None
    gap_agent = results[gap_idx]["agent"] if gap_idx is not None else None

    # Kneedle on deduped
    deduped = []
    for r in results:
        if deduped and float(r["similarity_score"]) == float(deduped[-1]["similarity_score"]):
            deduped[-1]["_agents"].append(r["agent"])
        else:
            deduped.append({**r, "_agents": [r["agent"]]})
    knee_idx = find_elbow(deduped) - 1
    knee_agent = deduped[knee_idx]["_agents"][-1] if knee_idx >= 0 else None

    cutoffs = []
    if gap_agent:  cutoffs.append({"name": "gap",  "agent": gap_agent})
    if knee_agent: cutoffs.append({"name": "knee", "agent": knee_agent})
    query["cutoffs"] = cutoffs

_save_queries(queries)
print("Done.")
