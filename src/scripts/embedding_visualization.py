"""
Embed all tables for every agent in an input file and visualize with UMAP + HDBSCAN and UMAP + EVoC.

Reads agent list from the input file, fetches embeddings (cached), and writes:
  - embedding_clusters_umap_hdbscan.png
  - embedding_clusters_umap_evoc.png
  - embedding_clusters_evoc.png             (EVoC node embedding 2D layout + EVoC clusters)
  - embedding_clusters_fixed_hdbscan.png    (fixed lon/lat coords, no UMAP)
  - embedding_clusters_fixed_evoc.png       (fixed lon/lat coords, no UMAP)
  - embedding_clusters.md
into the output directory.
"""

import concurrent.futures
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import fast_hdbscan
import matplotlib
import matplotlib.artist
import matplotlib.axes
import matplotlib.lines
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import umap

import qcp_adaptive_termination.agent_registry as agent_registry
from qcp_adaptive_termination.models import AgentRegistryType, DataAgent
from qcp_adaptive_termination.observability import context
from scripts.utils import (
    CATEGORICAL_COLORS,
    DB_GEO,
    NOISE_COLOR,
    ScriptContext,
    build_color_map,
    build_evoc_layout,
    run_script,
)

MIN_SAMPLES = 1
HDBSCAN_MIN_CLUSTER_SIZE = 5

_SVG_PATH = Path(__file__).parent / "res" / "Continents.svg"


def _color_for_label(label: int) -> str:
    if label == -1:
        return NOISE_COLOR
    return CATEGORICAL_COLORS[label % (len(CATEGORICAL_COLORS) - 1)]


def _db_label(agent: DataAgent) -> str:
    return Path(agent.path).stem


def _geo_coord(stem: str) -> tuple[float, float] | None:
    return DB_GEO.get(stem)


def _parse_svg_paths() -> list[matplotlib.path.Path]:
    """
    Parse Continents.svg and return matplotlib Path objects in SVG viewBox coordinates.
    Paths in Continents.svg are already in viewBox space (0 0 2752.766 1537.631) with no
    group transforms, so coordinates are used as-is.
    """
    if not _SVG_PATH.exists():
        return []

    tree = ET.parse(_SVG_PATH)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    paths: list[matplotlib.path.Path] = []

    for path_el in list(root.findall(".//svg:path", ns)) + list(root.findall(".//path")):
        if path_el.get("transform"):
            continue
        classes = path_el.get("class", "").split()
        if "ocean" in classes or "lake" in classes:
            continue
        d = path_el.get("d", "")
        if not d:
            continue
        try:
            mpl_path = _svg_d_to_mpl_path(d)
            if mpl_path is not None:
                paths.append(mpl_path)
        except Exception:
            continue

    return paths


def _svg_d_to_mpl_path(d: str) -> matplotlib.path.Path | None:
    """Convert SVG path d-string to a matplotlib Path in viewBox coordinates."""
    tokens = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", d)

    vertices: list[tuple[float, float]] = []
    codes: list[int] = []

    MOVETO = int(matplotlib.path.Path.MOVETO)
    LINETO = int(matplotlib.path.Path.LINETO)
    CURVE4 = int(matplotlib.path.Path.CURVE4)
    CLOSEPOLY = int(matplotlib.path.Path.CLOSEPOLY)

    i = 0
    cmd = "M"
    cx, cy = 0.0, 0.0

    def _next_floats(n: int) -> list[float]:
        nonlocal i
        vals = [float(tokens[i + k]) for k in range(n)]
        i += n
        return vals

    while i < len(tokens):
        tok = tokens[i]
        if tok.isalpha():
            cmd = tok
            i += 1
            continue

        if cmd in ("M", "m"):
            x, y = _next_floats(2)
            if cmd == "m":
                x, y = cx + x, cy + y
            cx, cy = x, y
            vertices.append((cx, cy))
            codes.append(MOVETO)
            cmd = "L" if cmd == "M" else "l"
        elif cmd in ("L", "l"):
            x, y = _next_floats(2)
            if cmd == "l":
                x, y = cx + x, cy + y
            cx, cy = x, y
            vertices.append((cx, cy))
            codes.append(LINETO)
        elif cmd in ("H", "h"):
            (x,) = _next_floats(1)
            if cmd == "h":
                x = cx + x
            cx = x
            vertices.append((cx, cy))
            codes.append(LINETO)
        elif cmd in ("V", "v"):
            (y,) = _next_floats(1)
            if cmd == "v":
                y = cy + y
            cy = y
            vertices.append((cx, cy))
            codes.append(LINETO)
        elif cmd in ("C", "c"):
            coords = _next_floats(6)
            if cmd == "c":
                coords = [
                    cx + coords[0],
                    cy + coords[1],
                    cx + coords[2],
                    cy + coords[3],
                    cx + coords[4],
                    cy + coords[5],
                ]
            for k in range(3):
                vertices.append((coords[2 * k], coords[2 * k + 1]))
                codes.append(CURVE4)
            cx, cy = coords[4], coords[5]
        elif cmd in ("S", "s"):
            coords = _next_floats(4)
            if cmd == "s":
                coords = [
                    cx + coords[0],
                    cy + coords[1],
                    cx + coords[2],
                    cy + coords[3],
                ]
            cp1 = (
                (2 * cx - vertices[-2][0], 2 * cy - vertices[-2][1])
                if codes and codes[-1] == CURVE4 and len(vertices) >= 2
                else (cx, cy)
            )
            for pt in [cp1, (coords[0], coords[1]), (coords[2], coords[3])]:
                vertices.append(pt)
                codes.append(CURVE4)
            cx, cy = coords[2], coords[3]
        elif cmd in ("Z", "z"):
            if vertices:
                vertices.append(vertices[-1])
                codes.append(CLOSEPOLY)
        else:
            if i < len(tokens) and not tokens[i].isalpha():
                i += 1
            else:
                i += 1
            continue

    if len(vertices) < 2:
        return None

    return matplotlib.path.Path(np.array(vertices), np.array(codes, dtype=np.uint8))


def _render_plot(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    agent_labels: list[str],
    color_fn: Callable[[int], str],
    label_fn: Callable[[int], str],
    title: str,
    plot_path: Path,
    svg_bg: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    if svg_bg:
        _draw_svg_background(ax)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude", fontsize=9, color="#52514e")
        ax.set_ylabel("Latitude", fontsize=9, color="#52514e")

    scatter_handles: dict[int, matplotlib.artist.Artist] = {}
    for i, (x, y) in enumerate(coords.tolist()):
        lbl = int(cluster_labels[i])
        color = color_fn(lbl)
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
        legend_elements.append(
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_fn(lbl),
                markersize=7,
                label=label_fn(lbl),
            )
        )

    ax.legend(handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.85, edgecolor="#cccccc")
    ax.set_title(title, fontsize=12, color="#0b0b0b")
    if not svg_bg:
        ax.set_xlabel("UMAP-1", fontsize=9, color="#52514e")
        ax.set_ylabel("UMAP-2", fontsize=9, color="#52514e")
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e0e0e0")

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {plot_path}")


def _draw_svg_background(ax: matplotlib.axes.Axes) -> None:
    """
    Draw Continents.svg as a light-gray background in lon/lat space [-180,180] x [-90,90].
    SVG viewBox 0 0 2752.766 1537.631 maps directly to lon/lat (equirectangular projection):
    x=0..2752.766 -> lon=-180..180, y=0..1537.631 -> lat=90..-90.
    """
    svg_paths = _parse_svg_paths()
    if not svg_paths:
        return

    # Map content bounds derived from linear fit of known country centroids (R²=0.999).
    # Equirectangular but with border padding — actual geographic extent is smaller than viewBox.
    mx0, mx1 = -109.73, 2651.67
    my0, my1 = 40.94, 1484.71

    def _remap(verts: np.ndarray) -> np.ndarray:
        out = verts.copy()
        out[:, 0] = -180.0 + (verts[:, 0] - mx0) / (mx1 - mx0) * 360.0
        out[:, 1] = 90.0 - (verts[:, 1] - my0) / (my1 - my0) * 180.0
        return out

    ax.set_facecolor("#ffffff")  # ocean = white
    for svg_path in svg_paths:
        remapped_verts = _remap(np.asarray(svg_path.vertices))
        remapped_path = matplotlib.path.Path(remapped_verts, svg_path.codes)
        patch = matplotlib.patches.PathPatch(
            remapped_path,
            facecolor="#b9b9b9",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.7,
            zorder=1,
        )
        ax.add_patch(patch)


def _cluster_summary(cluster_labels: np.ndarray, agent_labels: list[str]) -> dict[int, list[str]]:
    cluster_to_dbs: dict[int, list[str]] = {}
    for i, lbl in enumerate(cluster_labels):
        cluster_to_dbs.setdefault(int(lbl), []).append(agent_labels[i])
    return cluster_to_dbs


def _build_geo_init(agent_labels: list[str], n_components: int = 2) -> tuple[np.ndarray, int] | None:
    """
    Build a geo-biased UMAP init array where every DB stem starts at its geographic position.
    Non-matched stems (unknown DBs) start at zero — UMAP optimizes from there.
    Returns (init_array, n_pinned) or None if no DBs were matched.
    """
    n = len(agent_labels)
    custom_init = np.zeros((n, n_components), dtype=float)

    n_pinned = 0
    for i, label in enumerate(agent_labels):
        geo = _geo_coord(label)
        if geo is not None:
            lon, lat = geo
            # normalize lon [-180,180] → [-1,1], lat [-90,90] → [-1,1]
            custom_init[i, 0] = lon / 180.0
            custom_init[i, 1] = lat / 90.0
            n_pinned += 1

    if n_pinned == 0:
        return None
    return custom_init, n_pinned


@dataclass(slots=True)
class _Embeddings:
    agent_names: list[str]
    agent_labels: list[str]
    X: np.ndarray


@dataclass(slots=True)
class _HdbscanResult:
    labels: np.ndarray
    n_clusters: int
    n_noise: int


@dataclass(slots=True)
class _EvocResult:
    dr_coords: np.ndarray
    top_labels: np.ndarray
    fine_labels: np.ndarray
    color_map: dict[int, str]
    n_clusters: int
    n_fine_clusters: int
    n_noise: int


async def _load_embeddings(ctx: ScriptContext, registry_type: AgentRegistryType) -> _Embeddings:
    print(f"Loading embeddings for {len(ctx.data_agents)} agents …")
    all_table_embeddings: dict[str, dict[str, list[float]]] = await agent_registry.embedding_vectors_for_all_tables(
        ctx.data_agents, registry_type
    )
    agent_names: list[str] = []
    agent_labels: list[str] = []
    vectors: list[list[float]] = []
    for agent in ctx.data_agents:
        tables = all_table_embeddings.get(agent.name, {})
        if not tables:
            continue
        mat = np.array(list(tables.values()), dtype=float)
        agent_names.append(agent.name)
        agent_labels.append(_db_label(agent))
        vectors.append(mat.mean(axis=0).tolist())
    X = np.array(vectors, dtype=float)
    print(f"Embedding matrix: {X.shape}")
    return _Embeddings(agent_names=agent_names, agent_labels=agent_labels, X=X)


def _compute_hdbscan(X: np.ndarray) -> _HdbscanResult:
    hdb = fast_hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
    ).fit(X)
    labels: np.ndarray = cast(np.ndarray, hdb.labels_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE})")
    return _HdbscanResult(labels=labels, n_clusters=n_clusters, n_noise=n_noise)


def _compute_evoc(X: np.ndarray) -> _EvocResult:
    dr_coords, clusterer = build_evoc_layout(X)
    top_labels: np.ndarray = cast(np.ndarray, clusterer.labels_)
    # finest layer for coloring; falls back to top labels if only one layer
    fine_labels: np.ndarray = clusterer.cluster_layers_[0] if len(clusterer.cluster_layers_) > 1 else top_labels
    n_clusters = len(set(top_labels)) - (1 if -1 in top_labels else 0)
    n_fine_clusters = len(set(fine_labels)) - (1 if -1 in fine_labels else 0)
    n_noise = int((fine_labels == -1).sum())
    print(f"EVoC: {n_clusters} top-level clusters, {n_fine_clusters} fine clusters, {n_noise} noise points")
    print(f"EVoC DR: 2D node embedding shape {dr_coords.shape}")
    color_map = build_color_map(clusterer.cluster_layers_, top_labels)
    return _EvocResult(
        dr_coords=dr_coords,
        top_labels=top_labels,
        fine_labels=fine_labels,
        color_map=color_map,
        n_clusters=n_clusters,
        n_fine_clusters=n_fine_clusters,
        n_noise=n_noise,
    )


def _build_plot_jobs(
    emb: _Embeddings,
    umap_coords: np.ndarray,
    hdb: _HdbscanResult,
    evoc: _EvocResult,
    title_prefix: str,
    output_dir: Path,
) -> list[tuple]:
    def _evoc_color(fine_lbl: int) -> str:
        if fine_lbl == -1:
            return NOISE_COLOR
        return evoc.color_map.get(fine_lbl, NOISE_COLOR)

    def _evoc_legend_label(fine_lbl: int) -> str:
        if fine_lbl == -1:
            return "Noise"
        top = int(evoc.top_labels[np.where(evoc.fine_labels == fine_lbl)[0][0]])
        return f"Cluster {top}.{fine_lbl}"

    plot_jobs: list[tuple] = [
        (
            umap_coords,
            hdb.labels,
            emb.agent_labels,
            _color_for_label,
            lambda lbl: "Noise" if lbl == -1 else f"Cluster {lbl}",
            f"{title_prefix}DB Agent Embeddings — UMAP + HDBSCAN clusters",
            output_dir / "embedding_clusters_umap_hdbscan.png",
            False,
        ),
        (
            umap_coords,
            evoc.fine_labels,
            emb.agent_labels,
            _evoc_color,
            _evoc_legend_label,
            f"{title_prefix}DB Agent Embeddings — UMAP + EVoC clusters (hierarchical)",
            output_dir / "embedding_clusters_umap_evoc.png",
            False,
        ),
        (
            evoc.dr_coords,
            evoc.fine_labels,
            emb.agent_labels,
            _evoc_color,
            _evoc_legend_label,
            f"{title_prefix}DB Agent Embeddings — EVoC node embedding + EVoC clusters (hierarchical)",
            output_dir / "embedding_clusters_evoc.png",
            False,
        ),
    ]

    geo_result = _build_geo_init(emb.agent_labels)
    if geo_result is not None:
        fixed_coords = geo_result[0].copy().astype(float)
        fixed_coords[:, 0] *= 180.0
        fixed_coords[:, 1] *= 90.0
        plot_jobs += [
            (
                fixed_coords,
                hdb.labels,
                emb.agent_labels,
                _color_for_label,
                lambda lbl: "Noise" if lbl == -1 else f"Cluster {lbl}",
                f"{title_prefix}DB Agent Embeddings — Fixed lon/lat + HDBSCAN clusters",
                output_dir / "embedding_clusters_fixed_hdbscan.png",
                True,
            ),
            (
                fixed_coords,
                evoc.fine_labels,
                emb.agent_labels,
                _evoc_color,
                _evoc_legend_label,
                f"{title_prefix}DB Agent Embeddings — Fixed lon/lat + EVoC clusters (hierarchical)",
                output_dir / "embedding_clusters_fixed_evoc.png",
                True,
            ),
        ]

    return plot_jobs


def _render_plots(plot_jobs: list[tuple]) -> None:
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = [pool.submit(_render_plot, *job) for job in plot_jobs]
        for f in concurrent.futures.as_completed(futures):
            f.result()


def _md_cluster_section(cluster_map: dict[int, list[str]]) -> list[str]:
    section: list[str] = []
    for lbl in sorted(cluster_map.keys()):
        dbs = sorted(cluster_map[lbl])
        heading = "## Noise" if lbl == -1 else f"## Cluster {lbl}"
        section.append(heading)
        section.append(f"_{len(dbs)} databases_")
        section.append("")
        for db_name in dbs:
            section.append(f"- {db_name}")
        section.append("")
    return section


def _write_markdown_report(
    emb: _Embeddings,
    hdb: _HdbscanResult,
    evoc: _EvocResult,
    output_dir: Path,
) -> None:
    hdbscan_cluster_map = _cluster_summary(hdb.labels, emb.agent_labels)
    evoc_cluster_map = _cluster_summary(evoc.top_labels, emb.agent_labels)

    md_lines = [
        "# DB Embedding Clusters",
        "",
        f"**Total agents:** {len(emb.agent_names)}  ",
        "",
        "## HDBSCAN",
        "",
        f"**min_cluster_size:** {HDBSCAN_MIN_CLUSTER_SIZE}, min_samples: {MIN_SAMPLES}  ",
        f"**Clusters:** {hdb.n_clusters}  ",
        f"**Noise points:** {hdb.n_noise}",
        "",
        "![HDBSCAN visualization](embedding_clusters_umap_hdbscan.png)",
        "",
        *_md_cluster_section(hdbscan_cluster_map),
        "## EVoC",
        "",
        f"**min_samples:** {MIN_SAMPLES}  ",
        f"**Top-level clusters:** {evoc.n_clusters}  ",
        f"**Fine clusters:** {evoc.n_fine_clusters}  ",
        f"**Noise points:** {evoc.n_noise}",
        "",
        "![EVoC visualization](embedding_clusters_umap_evoc.png)",
        "",
        "![EVoC DR visualization](embedding_clusters_evoc.png)",
        "",
        *_md_cluster_section(evoc_cluster_map),
    ]

    md_path = output_dir / "embedding_clusters.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Report saved: {md_path}")


async def _run(ctx: ScriptContext) -> None:
    output_dir = context.get_output_dir()
    registry_type = AgentRegistryType(ctx.config.selection_kwargs.get("registry_type", AgentRegistryType.TECHNICAL))
    assert output_dir is not None

    embeddings = await _load_embeddings(ctx, registry_type)
    umap_coords = cast(np.ndarray, umap.UMAP(random_state=42, n_jobs=1).fit_transform(embeddings.X))
    hdb = _compute_hdbscan(umap_coords)
    evoc = _compute_evoc(embeddings.X)

    title_prefix = f"[{ctx.name} / {registry_type}] " if ctx.name else f"[{registry_type}] "
    plot_jobs = _build_plot_jobs(embeddings, umap_coords, hdb, evoc, title_prefix, output_dir)
    _render_plots(plot_jobs)
    _write_markdown_report(embeddings, hdb, evoc, output_dir)


if __name__ == "__main__":
    run_script(description="Visualize agent embeddings via UMAP + HDBSCAN and EVoC", run_fn=_run)
