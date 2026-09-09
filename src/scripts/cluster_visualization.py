"""
Visualize DBSCAN clusters of agent embeddings for a single query.

Shows all agents as UMAP-reduced 2D points, colored by DBSCAN cluster,
with the query point marked. Also writes a detailed log file with:
- relevant agents (from input file ground truth)
- per-cluster: agent names, L2 distances to query, which are relevant

Usage:
    python -m scripts.cluster_visualization
"""

import argparse
import asyncio
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cluster import DBSCAN

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
from qcp_adaptive_termination.models import AgentRegistryType, DataAgent
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.pipeline import load_config, load_input
from qcp_adaptive_termination.stages.discovery.pre_filter import DBSCAN_EPS, DBSCAN_MIN_SAMPLES

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1


def _color_for_label(label: int) -> str:
    if label == -1:
        return "#aaaaaa"
    cmap = plt.get_cmap("tab20")
    return cmap(label % 20)


async def _run(
    query: str,
    data_agents: list[DataAgent],
    relevant_agent_names: set[str],
    registry_type: AgentRegistryType,
    output_path: Path,
    log_path: Path,
) -> None:
    print(f"Fetching embeddings for {len(data_agents)} agents...")
    table_vectors = await agent_registry.embedding_vectors_for_all_tables(data_agents, registry_type)

    [query_emb] = await embedding.create([query])
    query_vec = np.array(query_emb, dtype=np.float32)

    # best table vector + distance per agent
    agent_names: list[str] = []
    agent_vecs: list[list[float]] = []
    agent_best_table: dict[str, str] = {}
    agent_best_dist: dict[str, float] = {}

    agent_map = {a.name: a for a in data_agents}
    for agent in data_agents:
        vecs = table_vectors.get(agent.name, {})
        if not vecs:
            continue
        best_table, best_vec = min(
            vecs.items(),
            key=lambda kv: float(np.linalg.norm(np.array(kv[1]) - query_vec)),
        )
        dist = float(np.linalg.norm(np.array(best_vec) - query_vec))
        agent_names.append(agent.name)
        agent_vecs.append(best_vec)
        agent_best_table[agent.name] = best_table
        agent_best_dist[agent.name] = dist

    print(f"Running DBSCAN on {len(agent_names)} agents (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
    matrix = np.array(agent_vecs, dtype=np.float32)
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="euclidean", n_jobs=-1).fit_predict(matrix)

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

    # build per-cluster info
    cluster_info: dict[int, dict] = {}
    for label in sorted(set(labels)):
        idxs = [i for i, l in enumerate(labels) if l == label]
        members = [agent_names[i] for i in idxs]
        centroid = matrix[idxs].mean(axis=0) if label != -1 else None
        centroid_dist = float(np.linalg.norm(centroid - query_vec)) if centroid is not None else None
        relevant_in_cluster = [m for m in members if m in relevant_agent_names]
        cluster_info[label] = {
            "members": members,
            "centroid_dist": centroid_dist,
            "relevant": relevant_in_cluster,
        }

    # write log
    lines: list[str] = []
    lines.append(f"Query: {query}")
    lines.append(f"Registry type: {registry_type}")
    lines.append(f"DBSCAN eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}")
    lines.append(f"Total agents: {len(agent_names)} | Clusters: {n_clusters} | Noise: {n_noise}")
    lines.append(f"Relevant agents (ground truth): {len(relevant_agent_names)}")
    lines.append("")

    # relevant agents not in any cluster (noise)
    relevant_noise = [n for n, l in zip(agent_names, labels) if l == -1 and n in relevant_agent_names]
    lines.append(f"=== NOISE ({n_noise} agents, {len(relevant_noise)} relevant) ===")
    for name in sorted([n for n, l in zip(agent_names, labels) if l == -1]):
        rel = " *** RELEVANT ***" if name in relevant_agent_names else ""
        path = agent_map[name].path if name in agent_map else "?"
        lines.append(f"  {name} ({path}) | L2={agent_best_dist[name]:.4f} | best_table={agent_best_table[name]}{rel}")
    lines.append("")

    # clusters sorted by centroid distance
    sorted_clusters = sorted(
        [(k, v) for k, v in cluster_info.items() if k != -1],
        key=lambda x: x[1]["centroid_dist"],
    )
    for label, info in sorted_clusters:
        n_rel = len(info["relevant"])
        lines.append(
            f"=== CLUSTER {label} ({len(info['members'])} agents, {n_rel} relevant | centroid dist={info['centroid_dist']:.4f}) ==="
        )
        for name in sorted(info["members"], key=lambda n: agent_best_dist[n]):
            rel = " *** RELEVANT ***" if name in relevant_agent_names else ""
            path = agent_map[name].path if name in agent_map else "?"
            lines.append(f"  {name} ({path}) | L2={agent_best_dist[name]:.4f} | best_table={agent_best_table[name]}{rel}")
        lines.append("")

    # summary: which relevant agents are in which cluster
    lines.append("=== RELEVANT AGENTS SUMMARY ===")
    for name in sorted(relevant_agent_names):
        if name not in agent_best_dist:
            lines.append(f"  {name}: no embedding found")
            continue
        idx = agent_names.index(name)
        label = labels[idx]
        cluster_str = f"CLUSTER {label}" if label != -1 else "NOISE"
        path = agent_map[name].path if name in agent_map else "?"
        lines.append(f"  {name} ({path}) | {cluster_str} | L2={agent_best_dist[name]:.4f} | best_table={agent_best_table[name]}")

    log_path.write_text("\n".join(lines))
    print(f"Log written to {log_path}")

    # plot
    print("Running UMAP...")
    all_vecs = np.vstack([matrix, query_vec.reshape(1, -1)])
    reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=42)
    reduced = reducer.fit_transform(all_vecs)

    agent_2d = reduced[:-1]
    query_2d = reduced[-1]

    fig, ax = plt.subplots(figsize=(14, 10))

    for i, (name, label) in enumerate(zip(agent_names, labels)):
        color = _color_for_label(label)
        is_relevant = name in relevant_agent_names
        marker = "x" if label == -1 else ("*" if is_relevant else "o")
        size = 80 if is_relevant else 20
        zorder = 4 if is_relevant else 2
        ax.scatter(agent_2d[i, 0], agent_2d[i, 1], c=[color], marker=marker, s=size, alpha=0.9 if is_relevant else 0.5, zorder=zorder)

    ax.scatter(query_2d[0], query_2d[1], c="red", marker="*", s=400, zorder=6, label="Query")

    # cluster centroids
    for label, info in cluster_info.items():
        if label == -1:
            continue
        idxs = [i for i, l in enumerate(labels) if l == label]
        centroid_2d = agent_2d[idxs].mean(axis=0)
        color = _color_for_label(label)
        ax.scatter(centroid_2d[0], centroid_2d[1], c=[color], marker="D", s=80, edgecolors="black", linewidths=1, zorder=5)
        ax.annotate(
            f"C{label} ({len(info['relevant'])}/{len(info['members'])})",
            centroid_2d, fontsize=6, ha="center", va="bottom",
        )

    # legend entries
    ax.scatter([], [], c="black", marker="*", s=80, label="Relevant agent")
    ax.scatter([], [], c="#aaaaaa", marker="x", s=40, label="Noise")

    ax.set_title(
        f"DBSCAN clusters — query: {query[:80]}\n"
        f"{n_clusters} clusters, {n_noise} noise, {len(agent_names)} agents | "
        f"★ = relevant (ground truth)"
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DBSCAN agent clusters for a query")
    parser.add_argument("--input", default="/mnt/labstore/nzeitz/dataset/wikidata-federated-overlap-dataset/input/How_many_airlines_are_headquartered_in_UK.json", help="Input JSON file with agents")
    parser.add_argument("--config", default="config.yaml", help="Config YAML")
    parser.add_argument("--data-dir", default="/mnt/labstore/nzeitz/dataset/wikidata-federated-overlap-dataset", help="Data directory")
    parser.add_argument("--query", default=None, help="Query string to embed (default: read from input file)")
    parser.add_argument("--cache_dir", default=None, help="Cache directory")
    parser.add_argument("--output", default="cluster_visualization.png", help="Output PNG path")
    parser.add_argument("--log", default="cluster_visualization.log", help="Output log path")
    parser.add_argument(
        "--registry-type",
        default="generated_nl",
        choices=["technical", "natural_language", "generated_nl"],
        help="Registry type for embeddings",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    input_file = Path(args.input)
    data_dir = Path(args.data_dir) if args.data_dir else input_file.parent
    registry_type = AgentRegistryType(args.registry_type)

    data, extra = load_input(input_file)
    data_agents: list[DataAgent] = data.data_agents  # type: ignore[union-attr]
    relevant_agent_names: set[str] = set(extra.get("relevant_data_agents", [])) if extra else set()

    query = args.query
    if query is None:
        query = data.query  # type: ignore[union-attr]
        print(f"Using query from input file: {query}")

    async def _runner() -> None:
        with context.pipeline_context(
            name="cluster-viz",
            trace_name="cluster-viz",
            output_dir=None,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            data_dir=data_dir,
            cache_mode=config.cache_mode,
            config=config,
            trace_metadata={},
        ):
            await _run(query, data_agents, relevant_agent_names, registry_type, Path(args.output), Path(args.log))

    asyncio.run(_runner())


if __name__ == "__main__":
    main()
