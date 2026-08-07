"""
Load all DB agent embeddings (from cache), cluster with DBSCAN, visualize with UMAP.

Outputs:
  - ../logs/embedding_clusters.png
  - ../logs/embedding_clusters.md
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.artist
import matplotlib.lines

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

import umap

from common.initialization import initialize
from common.agent_registry.agent_registry import AgentRegistry
from model.agent.data_agent import DataAgent
from model.execution_config import ExecutionConfig


INPUT_FILE = Path("../data/wikidata/companies/split_overlap_employees/employees.json")
OUTPUT_DIR = Path("../logs")
EPS = 0.062
MIN_SAMPLES = 3
DBSCAN_METRIC = "cosine"

CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#b55c1a",  # brown
    "#888888",  # noise / cluster -1
]

NOISE_COLOR = "#aaaaaa"


def color_for_label(label: int) -> str:
    if label == -1:
        return NOISE_COLOR
    return CATEGORICAL_COLORS[label % (len(CATEGORICAL_COLORS) - 1)]


async def load_embeddings(
    agents: list[DataAgent],
) -> dict[str, dict[str, list[float]]]:
    return await AgentRegistry.embedding_vectors_for_all_tables(agents)


def db_label(agent: DataAgent) -> str:
    return Path(agent.path).stem


async def run() -> None:
    initialize(log_name="visualize_embeddings", disable_cache=False)

    with open(INPUT_FILE) as f:
        data = json.load(f)

    raw_agents = data["value"]["data_agents"]
    agents = [DataAgent(**a) for a in raw_agents]

    ExecutionConfig.set(
        data_path_prefix="../data/wikidata/companies/split_overlap_employees/"
    )

    print(f"Loading embeddings for {len(agents)} agents …")
    all_table_embeddings: dict[str, dict[str, list[float]]] = await load_embeddings(
        agents
    )

    # For each agent use the mean of its table embeddings as the agent vector.
    agent_names: list[str] = []
    agent_labels: list[str] = []
    vectors: list[list[float]] = []

    for agent in agents:
        tables = all_table_embeddings.get(agent.name, {})
        if not tables:
            continue
        mat = np.array(list(tables.values()), dtype=float)
        vec = mat.mean(axis=0).tolist()
        agent_names.append(agent.name)
        agent_labels.append(db_label(agent))
        vectors.append(vec)

    X = np.array(vectors, dtype=float)
    print(f"Embedding matrix: {X.shape}")

    # DBSCAN clustering
    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric=DBSCAN_METRIC).fit(X)
    cluster_labels: np.ndarray = db.labels_
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int((cluster_labels == -1).sum())
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

    # UMAP reduction
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords: np.ndarray = cast(np.ndarray, reducer.fit_transform(X))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    scatter_handles: dict[int, matplotlib.artist.Artist] = {}
    for i, (x, y) in enumerate(coords.tolist()):
        lbl = int(cluster_labels[i])
        color = color_for_label(lbl)
        sc = ax.scatter(
            x, y, color=color, s=60, linewidths=0.5, edgecolors="white", zorder=3
        )
        if lbl not in scatter_handles:
            scatter_handles[lbl] = sc

        ax.annotate(
            agent_labels[i],
            (x, y),
            fontsize=5.5,
            color="#52514e",
            xytext=(3, 3),
            textcoords="offset points",
            clip_on=True,
        )

    # Legend: one entry per cluster
    legend_elements = []
    for lbl in sorted(scatter_handles.keys()):
        label_text = "Noise" if lbl == -1 else f"Cluster {lbl}"
        legend_elements.append(
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_for_label(lbl),
                markersize=7,
                label=label_text,
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=7,
        framealpha=0.85,
        edgecolor="#cccccc",
    )

    ax.set_title(
        "DB Agent Embeddings — UMAP + DBSCAN clusters", fontsize=12, color="#0b0b0b"
    )
    ax.set_xlabel("UMAP-1", fontsize=9, color="#52514e")
    ax.set_ylabel("UMAP-2", fontsize=9, color="#52514e")
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e0e0e0")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "embedding_clusters.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {plot_path}")

    # --- Markdown report ---
    cluster_to_dbs: dict[int, list[str]] = {}
    for i, lbl in enumerate(cluster_labels):
        cluster_to_dbs.setdefault(int(lbl), []).append(agent_labels[i])

    lines: list[str] = [
        "# DB Embedding Clusters",
        "",
        f"**Total agents:** {len(agent_names)}  ",
        f"**DBSCAN eps:** {EPS}, min_samples: {MIN_SAMPLES}, metric: {DBSCAN_METRIC}  ",
        f"**Clusters:** {n_clusters}  ",
        f"**Noise points:** {n_noise}",
        "",
        "![UMAP visualization](embedding_clusters.png)",
        "",
    ]

    for lbl in sorted(cluster_to_dbs.keys()):
        dbs = sorted(cluster_to_dbs[lbl])
        heading = "## Noise" if lbl == -1 else f"## Cluster {lbl}"
        lines.append(heading)
        lines.append(f"_{len(dbs)} databases_")
        lines.append("")
        for db_name in dbs:
            lines.append(f"- {db_name}")
        lines.append("")

    md_path = OUTPUT_DIR / "embedding_clusters.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {md_path}")


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
