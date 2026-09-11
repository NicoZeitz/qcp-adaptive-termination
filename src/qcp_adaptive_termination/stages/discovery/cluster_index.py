"""HDBSCAN-based cluster index for embedding pre-filter.

Clusters are built over individual table embeddings (not agent averages).
At query time, the closest table-clusters are selected via largest-gap cutoff,
and all agents that have at least one table in a selected cluster are kept.
"""

import logging
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


class ClusterIndex:
    """Cluster index built over table embeddings."""

    def __init__(
        self,
        *,
        # per-table data
        table_labels: list[str],          # "agent_name/table_name" for each row
        agent_per_table: list[str],        # agent_name for each row
        labels: list[int],                 # HDBSCAN cluster id per row
        # per-cluster data
        cluster_ids: list[int],
        centroids: np.ndarray,             # shape (n_clusters, dim)
        centroid_representatives: dict[int, str],  # cluster_id -> table_label
        cluster_to_tables: dict[int, list[str]],   # cluster_id -> [table_labels]
        cluster_to_agents: dict[int, list[str]],   # cluster_id -> unique agent names
        # 2D projection
        X_2d: np.ndarray,                  # shape (n_tables, 2)
    ) -> None:
        self.table_labels = table_labels
        self.agent_per_table = agent_per_table
        self.labels = labels
        self.cluster_ids = cluster_ids
        self.centroids = centroids
        self.centroid_representatives = centroid_representatives
        self.cluster_to_tables = cluster_to_tables
        self.cluster_to_agents = cluster_to_agents
        self.X_2d = X_2d


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


def _build_index(
    table_labels: list[str],
    agent_per_table: list[str],
    X: np.ndarray,
) -> ClusterIndex:
    n = len(table_labels)
    min_cluster_size = max(5, n // 200)
    logging.info("ClusterIndex: HDBSCAN over %d tables (min_cluster_size=%d)…", n, min_cluster_size)

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

    cluster_ids = sorted(set(labels.tolist()))
    cluster_to_tables: dict[int, list[str]] = {c: [] for c in cluster_ids}
    cluster_to_agents: dict[int, list[str]] = {c: [] for c in cluster_ids}
    seen_agents: dict[int, set[str]] = {c: set() for c in cluster_ids}
    for i, lbl in enumerate(labels.tolist()):
        cluster_to_tables[lbl].append(table_labels[i])
        ag = agent_per_table[i]
        if ag not in seen_agents[lbl]:
            seen_agents[lbl].add(ag)
            cluster_to_agents[lbl].append(ag)

    centroids = np.array([X[labels == c].mean(axis=0) for c in cluster_ids], dtype=np.float32)

    centroid_representatives: dict[int, str] = {}
    for ci, c in enumerate(cluster_ids):
        idxs = np.where(labels == c)[0]
        dists = np.linalg.norm(X[idxs] - centroids[ci], axis=1)
        centroid_representatives[c] = table_labels[idxs[int(np.argmin(dists))]]

    logging.info("ClusterIndex: %d clusters built", len(cluster_ids))
    logging.info("ClusterIndex: computing UMAP 2D projection…")
    X_2d = np.array(umap.UMAP(n_neighbors=50, random_state=42, n_jobs=1).fit_transform(X), dtype=np.float32)

    return ClusterIndex(
        table_labels=table_labels,
        agent_per_table=agent_per_table,
        labels=labels.tolist(),
        cluster_ids=cluster_ids,
        centroids=centroids,
        centroid_representatives=centroid_representatives,
        cluster_to_tables=cluster_to_tables,
        cluster_to_agents=cluster_to_agents,
        X_2d=X_2d,
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
        return ClusterIndex(
            table_labels=v["table_labels"],
            agent_per_table=v["agent_per_table"],
            labels=v["labels"],
            cluster_ids=v["cluster_ids"],
            centroids=np.array(v["centroids"], dtype=np.float32),
            centroid_representatives={int(k): v2 for k, v2 in v["centroid_representatives"].items()},
            cluster_to_tables={int(k): v2 for k, v2 in v["cluster_to_tables"].items()},
            cluster_to_agents={int(k): v2 for k, v2 in v["cluster_to_agents"].items()},
            X_2d=np.array(v["X_2d"], dtype=np.float32),
        )
    except Exception as exc:
        logging.warning("ClusterIndex: failed to deserialize cache: %s", exc)
        return None


def _store_to_cache(key: dict, index: ClusterIndex) -> None:
    cache = get_cache()
    if cache is None:
        return
    value = {
        "table_labels": index.table_labels,
        "agent_per_table": index.agent_per_table,
        "labels": index.labels,
        "cluster_ids": index.cluster_ids,
        "centroids": index.centroids.tolist(),
        "centroid_representatives": {str(k): v for k, v in index.centroid_representatives.items()},
        "cluster_to_tables": {str(k): v for k, v in index.cluster_to_tables.items()},
        "cluster_to_agents": {str(k): v for k, v in index.cluster_to_agents.items()},
        "X_2d": index.X_2d.tolist(),
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

    table_labels: list[str] = []
    agent_per_table: list[str] = []
    vecs: list[list[float]] = []
    for agent in agents:
        for table_name, vec in table_vectors.get(agent.name, {}).items():
            table_labels.append(f"{agent.name}/{table_name}")
            agent_per_table.append(agent.name)
            vecs.append(vec)

    if not table_labels:
        logging.warning("ClusterIndex: no table embeddings found")
        all_names = [a.name for a in agents]
        dummy = np.zeros((1, 2), dtype=np.float32)
        return ClusterIndex(
            table_labels=[], agent_per_table=[], labels=[],
            cluster_ids=[0], centroids=dummy,
            centroid_representatives={},
            cluster_to_tables={0: []},
            cluster_to_agents={0: all_names},
            X_2d=dummy,
        )

    X = np.array(vecs, dtype=np.float32)
    index = _build_index(table_labels, agent_per_table, X)
    _store_to_cache(key, index)
    return index


def filter_agents_by_cluster(
    query_embedding: list[float],
    index: ClusterIndex,
    agents: list[DataAgent],
    find_l2_clusters_fn,
) -> tuple[list[DataAgent], set[int]]:
    """Return (filtered_agents, selected_cluster_ids)."""
    query = np.array(query_embedding, dtype=np.float32)
    dists = np.linalg.norm(index.centroids - query, axis=1).tolist()

    sorted_pairs = sorted(zip(dists, index.cluster_ids))
    sorted_dists = [d for d, _ in sorted_pairs]
    sorted_ids = [c for _, c in sorted_pairs]

    cutoff_idx = find_l2_clusters_fn(sorted_dists)
    selected_clusters = set(sorted_ids[:cutoff_idx])

    logging.info(
        "ClusterIndex: selected %d / %d clusters (gap at idx %d, dist %.4f → %.4f)",
        len(selected_clusters), len(index.cluster_ids), cutoff_idx,
        sorted_dists[cutoff_idx - 1] if cutoff_idx > 0 else 0.0,
        sorted_dists[cutoff_idx] if cutoff_idx < len(sorted_dists) else float("inf"),
    )
    for c in sorted_ids[:cutoff_idx]:
        rep = index.centroid_representatives.get(c, "?")
        idx = index.cluster_ids.index(c)
        logging.info("  → cluster %d (dist=%.4f, representative=%s)", c, dists[idx], rep)

    kept_agent_names: set[str] = set()
    for c in selected_clusters:
        kept_agent_names.update(index.cluster_to_agents.get(c, []))

    agent_map = {a.name: a for a in agents}
    return [agent_map[n] for n in kept_agent_names if n in agent_map], selected_clusters


def render_query_visualization(
    query: str,
    query_embedding: list[float],
    index: ClusterIndex,
    selected_clusters: set[int],
    output_path: Path,
) -> None:
    if not index.table_labels or index.X_2d.shape[0] == 0:
        return

    query_vec = np.array(query_embedding, dtype=np.float32)
    dists = np.linalg.norm(index.centroids - query_vec, axis=1)
    nearest_cluster = index.cluster_ids[int(np.argmin(dists))]

    # query 2D position: mean of nearest cluster's table coords
    nearest_idxs = [i for i, lbl in enumerate(index.labels) if lbl == nearest_cluster]
    query_2d = index.X_2d[nearest_idxs].mean(axis=0) if nearest_idxs else index.X_2d.mean(axis=0)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    for i, lbl in enumerate(index.labels):
        is_sel = lbl in selected_clusters
        color = _color_for_cluster(lbl) if is_sel else "#dddddd"
        ax.scatter(index.X_2d[i, 0], index.X_2d[i, 1], color=color, s=15,
                   alpha=0.7 if is_sel else 0.25, linewidths=0, zorder=2)

    legend_elements = []
    for ci, c in enumerate(index.cluster_ids):
        tidxs = [i for i, lbl in enumerate(index.labels) if lbl == c]
        if not tidxs:
            continue
        cx, cy = index.X_2d[tidxs].mean(axis=0)
        is_sel = c in selected_clusters
        color = _color_for_cluster(c) if is_sel else "#bbbbbb"
        ax.scatter(cx, cy, color=color, s=120, marker="D", edgecolors="white",
                   linewidths=1.0, zorder=4, alpha=1.0 if is_sel else 0.35)
        rep = index.centroid_representatives.get(c, "")
        if is_sel:
            ax.annotate(rep, (cx, cy), fontsize=5.5, color="#0b0b0b",
                        xytext=(4, 4), textcoords="offset points", zorder=5)
            legend_elements.append(
                matplotlib.lines.Line2D([0], [0], marker="D", color="w",
                                        markerfacecolor=color, markersize=7,
                                        label=f"C{c}: {rep}")
            )

    ax.scatter(query_2d[0], query_2d[1], color="#ff0000", s=220, marker="*",
               edgecolors="white", linewidths=1.5, zorder=6)
    ax.annotate(
        f"Q: {query[:60]}{'…' if len(query) > 60 else ''}",
        (query_2d[0], query_2d[1]), fontsize=6, color="#cc0000",
        xytext=(6, 6), textcoords="offset points", zorder=7,
    )

    legend_elements.insert(0, matplotlib.lines.Line2D(
        [0], [0], marker="*", color="w", markerfacecolor="#ff0000", markersize=10, label="Query"
    ))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=6,
              framealpha=0.85, edgecolor="#cccccc", ncol=2)
    ax.set_title(
        f"Cluster filter: {len(selected_clusters)}/{len(index.cluster_ids)} clusters selected",
        fontsize=11, color="#0b0b0b",
    )
    ax.set_xlabel("UMAP-1", fontsize=9, color="#52514e")
    ax.set_ylabel("UMAP-2", fontsize=9, color="#52514e")
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e0e0e0")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("ClusterIndex: query visualization saved: %s", output_path)
