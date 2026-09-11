"""HDBSCAN-based cluster index for embedding pre-filter.

Builds a cluster index over agent embeddings, caches it to disk, and provides
a largest-gap filter that reduces the agent pool to thematically relevant clusters
before the per-table similarity filter runs.

Also caches UMAP 2D coordinates alongside the cluster data and renders a per-query
visualization showing all agents, cluster centroids, and the query position.
"""

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import fast_hdbscan
import matplotlib
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import umap

import qcp_adaptive_termination.agent_registry as agent_registry
from qcp_adaptive_termination.models import AgentRegistryType, DataAgent
from qcp_adaptive_termination.observability.context import get_cache, get_output_dir

matplotlib.use("Agg")

_CACHE_NAME_PREFIX = "ClusterIndex"
_CATEGORICAL_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]
_NOISE_COLOR = "#cccccc"


@dataclass(slots=True)
class ClusterIndex:
    centroids: np.ndarray                    # shape (n_clusters, embedding_dim)
    cluster_to_agents: dict[int, list[str]]  # cluster_id -> agent names
    agent_to_cluster: dict[str, int]         # agent_name -> cluster_id
    agent_names: list[str]                   # ordered list matching X_2d rows
    X_2d: np.ndarray                         # shape (n_agents, 2), UMAP coords
    centroid_representatives: dict[int, str] = field(default_factory=dict)  # cluster_id -> agent name


def _assign_noise_to_nearest(labels: np.ndarray, X: np.ndarray) -> np.ndarray:
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels
    cluster_ids = sorted(set(labels[~noise_mask]))
    centroids = np.array([X[labels == c].mean(axis=0) for c in cluster_ids])
    noise_indices = np.where(noise_mask)[0]
    dists = np.linalg.norm(X[noise_indices][:, None] - centroids[None, :], axis=2)
    nearest = np.array([cluster_ids[i] for i in np.argmin(dists, axis=1)], dtype=np.int32)
    result = labels.copy()
    result[noise_indices] = nearest
    return result


def _color_for_cluster(cluster_id: int) -> str:
    return _CATEGORICAL_COLORS[cluster_id % len(_CATEGORICAL_COLORS)]


def _build_index(agent_names: list[str], X: np.ndarray) -> ClusterIndex:
    min_cluster_size = max(5, len(agent_names) // 200)
    logging.info("ClusterIndex: building HDBSCAN (min_cluster_size=%d) over %d agents", min_cluster_size, len(agent_names))

    hdb = fast_hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit(X)
    labels = np.array(hdb.labels_, dtype=np.int32)

    n_noise = int((labels == -1).sum())
    if n_noise > 0:
        logging.info("ClusterIndex: assigning %d noise points to nearest cluster", n_noise)
        labels = _assign_noise_to_nearest(labels, X)

    cluster_to_agents: dict[int, list[str]] = {}
    agent_to_cluster: dict[str, int] = {}
    for i, name in enumerate(agent_names):
        lbl = int(labels[i])
        cluster_to_agents.setdefault(lbl, []).append(name)
        agent_to_cluster[name] = lbl

    cluster_ids = list(cluster_to_agents.keys())
    centroids = np.array([X[labels == c].mean(axis=0) for c in cluster_ids], dtype=np.float32)

    # representative: agent closest to centroid per cluster
    centroid_representatives: dict[int, str] = {}
    for i, c in enumerate(cluster_ids):
        idxs = np.where(labels == c)[0]
        dists = np.linalg.norm(X[idxs] - centroids[i], axis=1)
        centroid_representatives[c] = agent_names[idxs[int(np.argmin(dists))]]

    logging.info("ClusterIndex: %d clusters built", len(cluster_ids))

    logging.info("ClusterIndex: computing UMAP 2D projection…")
    X_2d = np.array(umap.UMAP(n_neighbors=50, random_state=42, n_jobs=1).fit_transform(X), dtype=np.float32)

    return ClusterIndex(
        centroids=centroids,
        cluster_to_agents=cluster_to_agents,
        agent_to_cluster=agent_to_cluster,
        agent_names=agent_names,
        X_2d=X_2d,
        centroid_representatives=centroid_representatives,
    )


def _cache_key(agent_names: list[str], registry_type: AgentRegistryType) -> dict:
    return {"agents": sorted(agent_names), "registry_type": str(registry_type)}


def _load_from_cache(key: dict) -> ClusterIndex | None:
    cache = get_cache()
    if cache is None:
        return None
    value, found = cache.get(key, name_prefix=_CACHE_NAME_PREFIX)
    if not found or value is None:
        return None
    try:
        v = value  # type: ignore[assignment]
        centroids = np.array(v["centroids"], dtype=np.float32)
        cluster_to_agents = {int(k): v2 for k, v2 in v["cluster_to_agents"].items()}
        agent_to_cluster = {k: int(v2) for k, v2 in v["agent_to_cluster"].items()}
        agent_names = v["agent_names"]
        X_2d = np.array(v["X_2d"], dtype=np.float32)
        centroid_representatives = {int(k): v2 for k, v2 in v.get("centroid_representatives", {}).items()}
        logging.info("ClusterIndex: loaded from cache")
        return ClusterIndex(
            centroids=centroids,
            cluster_to_agents=cluster_to_agents,
            agent_to_cluster=agent_to_cluster,
            agent_names=agent_names,
            X_2d=X_2d,
            centroid_representatives=centroid_representatives,
        )
    except Exception as exc:
        logging.warning("ClusterIndex: failed to deserialize cache entry: %s", exc)
        return None


def _store_to_cache(key: dict, index: ClusterIndex) -> None:
    cache = get_cache()
    if cache is None:
        return
    value = {
        "centroids": index.centroids.tolist(),
        "cluster_to_agents": {str(k): v for k, v in index.cluster_to_agents.items()},
        "agent_to_cluster": index.agent_to_cluster,
        "agent_names": index.agent_names,
        "X_2d": index.X_2d.tolist(),
        "centroid_representatives": {str(k): v for k, v in index.centroid_representatives.items()},
    }
    cache.store(key, value, name_prefix=_CACHE_NAME_PREFIX)


async def get_or_build_cluster_index(
    agents: list[DataAgent],
    registry_type: AgentRegistryType,
) -> ClusterIndex:
    key = _cache_key([a.name for a in agents], registry_type)
    cached = _load_from_cache(key)
    if cached is not None:
        return cached

    table_vectors = await agent_registry.embedding_vectors_for_all_tables(agents, registry_type)

    agent_names: list[str] = []
    agent_vecs: list[np.ndarray] = []
    for agent in agents:
        tables = table_vectors.get(agent.name, {})
        if tables:
            vecs = np.array(list(tables.values()), dtype=np.float32)
            agent_names.append(agent.name)
            agent_vecs.append(vecs.mean(axis=0))

    if not agent_names:
        logging.warning("ClusterIndex: no embeddings found, returning trivial index")
        dummy_2d = np.zeros((len(agents), 2), dtype=np.float32)
        all_names = [a.name for a in agents]
        return ClusterIndex(
            centroids=np.zeros((1, 1), dtype=np.float32),
            cluster_to_agents={0: all_names},
            agent_to_cluster={n: 0 for n in all_names},
            agent_names=all_names,
            X_2d=dummy_2d,
            centroid_representatives={0: all_names[0]} if all_names else {},
        )

    X = np.array(agent_vecs, dtype=np.float32)
    index = _build_index(agent_names, X)
    _store_to_cache(key, index)
    return index


def filter_agents_by_cluster(
    query_embedding: list[float],
    index: ClusterIndex,
    agents: list[DataAgent],
    find_l2_clusters_fn,
) -> list[DataAgent]:
    query = np.array(query_embedding, dtype=np.float32)
    cluster_ids = list(index.cluster_to_agents.keys())
    dists = np.linalg.norm(index.centroids - query, axis=1).tolist()

    sorted_pairs = sorted(zip(dists, cluster_ids))
    sorted_dists = [d for d, _ in sorted_pairs]
    sorted_ids = [c for _, c in sorted_pairs]

    cutoff_idx = find_l2_clusters_fn(sorted_dists)
    selected_clusters = set(sorted_ids[:cutoff_idx])

    logging.info(
        "ClusterIndex: selected %d / %d clusters (gap at idx %d, dist %.4f → %.4f)",
        len(selected_clusters),
        len(cluster_ids),
        cutoff_idx,
        sorted_dists[cutoff_idx - 1] if cutoff_idx > 0 else 0.0,
        sorted_dists[cutoff_idx] if cutoff_idx < len(sorted_dists) else float("inf"),
    )
    for c in sorted_ids[:cutoff_idx]:
        rep = index.centroid_representatives.get(c, "?")
        logging.info("  → cluster %d (dist=%.4f, representative=%s)", c, dists[cluster_ids.index(c)], rep)

    kept_names = set()
    for c in selected_clusters:
        kept_names.update(index.cluster_to_agents.get(c, []))

    agent_map = {a.name: a for a in agents}
    return [agent_map[n] for n in kept_names if n in agent_map]


def render_query_visualization(
    query: str,
    query_embedding: list[float],
    index: ClusterIndex,
    selected_clusters: set[int],
    output_path: Path,
) -> None:
    """Render a 2D UMAP plot showing all agents, cluster centroids, and the query point."""
    if index.X_2d.shape[0] == 0:
        return

    # project query into 2D by finding nearest agent and using its coords as proxy
    query_vec = np.array(query_embedding, dtype=np.float32)
    agent_vecs_order = {n: i for i, n in enumerate(index.agent_names)}

    # nearest agent in embedding space = proxy position for query in 2D
    all_names = list(index.agent_to_cluster.keys())
    # compute distances from query to each agent in full embedding space
    # we only have centroids in full space, so use centroid as proxy for 2D query position
    cluster_ids = list(index.cluster_to_agents.keys())
    centroid_dists = np.linalg.norm(index.centroids - query_vec, axis=1)
    nearest_cluster_idx = int(np.argmin(centroid_dists))
    nearest_cluster = cluster_ids[nearest_cluster_idx]
    rep_name = index.centroid_representatives.get(nearest_cluster)
    if rep_name and rep_name in agent_vecs_order:
        rep_idx = agent_vecs_order[rep_name]
        query_2d = index.X_2d[rep_idx]
    else:
        query_2d = index.X_2d.mean(axis=0)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    # draw all agent points
    for i, name in enumerate(index.agent_names):
        c = index.agent_to_cluster.get(name, -1)
        is_selected = c in selected_clusters
        color = _color_for_cluster(c) if is_selected else "#dddddd"
        alpha = 0.85 if is_selected else 0.3
        ax.scatter(index.X_2d[i, 0], index.X_2d[i, 1], color=color, s=20, alpha=alpha,
                   linewidths=0, zorder=2)

    # draw cluster centroids in 2D (compute as mean of their agents' 2D coords)
    legend_elements = []
    for ci, c in enumerate(cluster_ids):
        agent_idxs = [agent_vecs_order[n] for n in index.cluster_to_agents.get(c, []) if n in agent_vecs_order]
        if not agent_idxs:
            continue
        cx, cy = index.X_2d[agent_idxs].mean(axis=0)
        is_selected = c in selected_clusters
        color = _color_for_cluster(c) if is_selected else "#bbbbbb"
        ax.scatter(cx, cy, color=color, s=120, marker="D", edgecolors="white",
                   linewidths=1.0, zorder=4, alpha=1.0 if is_selected else 0.4)
        rep = index.centroid_representatives.get(c, "")
        if is_selected:
            ax.annotate(rep, (cx, cy), fontsize=5.5, color="#0b0b0b",
                        xytext=(4, 4), textcoords="offset points", zorder=5)
            legend_elements.append(
                matplotlib.lines.Line2D([0], [0], marker="D", color="w",
                                        markerfacecolor=color, markersize=7,
                                        label=f"Cluster {c}: {rep}")
            )

    # draw query point
    ax.scatter(query_2d[0], query_2d[1], color="#ff0000", s=200, marker="*",
               edgecolors="white", linewidths=1.5, zorder=6, label="Query")
    ax.annotate(
        f"Q: {query[:60]}{'…' if len(query) > 60 else ''}",
        (query_2d[0], query_2d[1]),
        fontsize=6, color="#cc0000",
        xytext=(6, 6), textcoords="offset points", zorder=7,
    )

    legend_elements.insert(0, matplotlib.lines.Line2D(
        [0], [0], marker="*", color="w", markerfacecolor="#ff0000", markersize=10, label="Query"
    ))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=6, framealpha=0.85,
              edgecolor="#cccccc", ncol=2)
    ax.set_title(f"Cluster filter: {len(selected_clusters)}/{len(cluster_ids)} clusters selected",
                 fontsize=11, color="#0b0b0b")
    ax.set_xlabel("UMAP-1", fontsize=9, color="#52514e")
    ax.set_ylabel("UMAP-2", fontsize=9, color="#52514e")
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e0e0e0")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("ClusterIndex: query visualization saved: %s", output_path)
