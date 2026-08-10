"""
Embed all tables for every agent in an input file and visualize with UMAP + DBSCAN.

Reads agent list from the input file, fetches embeddings (cached), and writes:
  - embedding_clusters.png
  - embedding_clusters.md
into the output directory.
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.artist
import matplotlib.lines

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cluster import DBSCAN

import qcp_adaptive_termination.agent_registry as agent_registry
from qcp_adaptive_termination.models import AgentRegistryType, DataAgent
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.pipeline import load_config, load_input

EPS = 0.062
MIN_SAMPLES = 3
DBSCAN_METRIC = "cosine"

CATEGORICAL_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#b55c1a",
    "#888888",
]
NOISE_COLOR = "#aaaaaa"


def _color_for_label(label: int) -> str:
    if label == -1:
        return NOISE_COLOR
    return CATEGORICAL_COLORS[label % (len(CATEGORICAL_COLORS) - 1)]


def _db_label(agent: DataAgent) -> str:
    return Path(agent.path).stem


async def _run(
    agents: list[DataAgent],
    registry_type: AgentRegistryType,
) -> None:
    output_dir = context.get_output_dir()
    assert output_dir is not None
    print(f"Loading embeddings for {len(agents)} agents …")
    all_table_embeddings: dict[str, dict[str, list[float]]] = await agent_registry.embedding_vectors_for_all_tables(
        agents, registry_type
    )

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
        agent_labels.append(_db_label(agent))
        vectors.append(vec)

    X = np.array(vectors, dtype=float)
    print(f"Embedding matrix: {X.shape}")

    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric=DBSCAN_METRIC).fit(X)
    cluster_labels: np.ndarray = db.labels_
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int((cluster_labels == -1).sum())
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, n_jobs=1)
    coords = cast(np.ndarray, reducer.fit_transform(X))

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    scatter_handles: dict[int, matplotlib.artist.Artist] = {}
    for i, (x, y) in enumerate(coords.tolist()):
        lbl = int(cluster_labels[i])
        color = _color_for_label(lbl)
        sc = ax.scatter(x, y, color=color, s=60, linewidths=0.5, edgecolors="white", zorder=3)
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

    legend_elements = []
    for lbl in sorted(scatter_handles.keys()):
        label_text = "Noise" if lbl == -1 else f"Cluster {lbl}"
        legend_elements.append(
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=_color_for_label(lbl),
                markersize=7,
                label=label_text,
            )
        )

    ax.legend(handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.85, edgecolor="#cccccc")
    ax.set_title("DB Agent Embeddings — UMAP + DBSCAN clusters", fontsize=12, color="#0b0b0b")
    ax.set_xlabel("UMAP-1", fontsize=9, color="#52514e")
    ax.set_ylabel("UMAP-2", fontsize=9, color="#52514e")
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e0e0e0")

    plot_path = output_dir / "embedding_clusters.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {plot_path}")

    cluster_to_dbs: dict[int, list[str]] = {}
    for i, lbl in enumerate(cluster_labels):
        cluster_to_dbs.setdefault(int(lbl), []).append(agent_labels[i])

    lines = [
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

    md_path = output_dir / "embedding_clusters.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize agent embeddings via UMAP + DBSCAN")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for agent data files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots and reports")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--config", type=str, default=None, help="Config as YAML file path or inline YAML string")
    args = parser.parse_args()

    input_file = Path(args.input)
    data_dir = Path(args.data_dir) if args.data_dir else input_file.parent
    datetime_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    output_dir = Path(args.output_dir) / datetime_str if args.output_dir else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    config = load_config(args.config)
    registry_type = AgentRegistryType(config.selection_kwargs.get("registry_type", AgentRegistryType.TECHNICAL))

    data, _extra = load_input(input_file)
    agents: list[DataAgent] = data.data_agents

    with context.pipeline_context(
        name="create_embedding_visualization",
        output_dir=output_dir,
        cache_dir=cache_dir,
        data_dir=data_dir,
        config=config,
    ):
        asyncio.run(_run(agents, registry_type))


if __name__ == "__main__":
    main()
