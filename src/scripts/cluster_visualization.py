"""
Visualize DBSCAN clusters of agent embeddings for a single query.

Shows all agents as UMAP-reduced 2D points, colored by DBSCAN cluster,
with the query point marked. Helps diagnose whether clustering separates
relevant from irrelevant agents.

Usage:
    python -m scripts.cluster_visualization --input <input.json> --query "How many airlines..."
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


def _color_for_label(label: int, n_clusters: int) -> str:
    if label == -1:
        return "#aaaaaa"
    cmap = plt.get_cmap("tab20")
    return cmap(label % 20)


async def _run(query: str, data_agents: list[DataAgent], registry_type: AgentRegistryType, output_path: Path) -> None:
    print(f"Fetching embeddings for {len(data_agents)} agents...")
    table_vectors = await agent_registry.embedding_vectors_for_all_tables(data_agents, registry_type)

    # best table vector per agent
    agent_names: list[str] = []
    agent_vecs: list[list[float]] = []
    for agent in data_agents:
        vecs = table_vectors.get(agent.name, {})
        if not vecs:
            continue
        [query_emb] = await embedding.create([query])
        best_vec = min(vecs.values(), key=lambda v: float(np.linalg.norm(np.array(v) - np.array(query_emb))))
        agent_names.append(agent.name)
        agent_vecs.append(best_vec)

    print(f"Running DBSCAN on {len(agent_names)} agents (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
    matrix = np.array(agent_vecs, dtype=np.float32)

    [query_emb] = await embedding.create([query])
    query_vec = np.array(query_emb, dtype=np.float32)

    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="euclidean", n_jobs=-1).fit_predict(matrix)

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

    # cluster centroids and distances to query
    for label in sorted(set(labels) - {-1}):
        mask = labels == label
        centroid = matrix[mask].mean(axis=0)
        dist = float(np.linalg.norm(centroid - query_vec))
        members = [agent_names[i] for i, l in enumerate(labels) if l == label]
        print(f"  Cluster {label}: {len(members)} agents, centroid dist={dist:.4f}")

    print("Running UMAP...")
    all_vecs = np.vstack([matrix, query_vec.reshape(1, -1)])
    reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=42)
    reduced = reducer.fit_transform(all_vecs)

    agent_2d = reduced[:-1]
    query_2d = reduced[-1]

    fig, ax = plt.subplots(figsize=(14, 10))

    for i, (name, label) in enumerate(zip(agent_names, labels)):
        color = _color_for_label(label, n_clusters)
        marker = "x" if label == -1 else "o"
        ax.scatter(agent_2d[i, 0], agent_2d[i, 1], c=[color], marker=marker, s=30, alpha=0.7)

    ax.scatter(query_2d[0], query_2d[1], c="red", marker="*", s=300, zorder=5, label="Query")

    # cluster centroids in 2D (average of member points)
    for label in sorted(set(labels) - {-1}):
        mask = labels == label
        centroid_2d = agent_2d[mask].mean(axis=0)
        color = _color_for_label(label, n_clusters)
        ax.scatter(centroid_2d[0], centroid_2d[1], c=[color], marker="D", s=80, edgecolors="black", linewidths=1, zorder=4)
        ax.annotate(f"C{label}", centroid_2d, fontsize=7, ha="center", va="bottom")

    ax.set_title(f"DBSCAN clusters — query: {query[:80]}\n{n_clusters} clusters, {n_noise} noise, {len(agent_names)} agents")
    ax.legend()
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DBSCAN agent clusters for a query")
    parser.add_argument("--input", default="/mnt/labstore/nzeitz/dataset/wikidata-federated-overlap-dataset/input/How_many_airlines_are_headquartered_in_UK.json", help="Input JSON file with agents")
    parser.add_argument("--config", default="config.yaml", help="Config YAML")
    parser.add_argument("--data-dir", default="/mnt/labstore/nzeitz/dataset/wikidata-federated-overlap-dataset", help="Data directory")
    parser.add_argument("--query", default=None, help="Query string to embed (default: read from input file)")
    parser.add_argument("--output", default="cluster_visualization.png", help="Output PNG path")
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

    query = args.query
    if query is None:
        query = data.query  # type: ignore[union-attr]
        print(f"Using query from input file: {query}")

    async def _runner() -> None:
        with context.pipeline_context(
            name="cluster-viz",
            trace_name="cluster-viz",
            output_dir=None,
            cache_dir=None,
            data_dir=data_dir,
            cache_mode=config.cache_mode,
            config=config,
            trace_metadata={},
        ):
            await _run(args.query, data_agents, registry_type, Path(args.output))

    asyncio.run(_runner())


if __name__ == "__main__":
    main()
