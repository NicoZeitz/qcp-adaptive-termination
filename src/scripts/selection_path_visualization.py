"""
Visualize the adaptive selection path during multi-agent query plan execution.

For each MultiAgentQueryPlanStep, iterates over the adaptive batching rounds and
produces two geo-map plots per round:
  - variant_a: agent dot size = ScoredDataAgent score (higher = more relevant)
  - variant_b: agent dot size = ground-truth new-row count (higher = more data)

Already-queried agents appear in gray at constant size.
Not-yet-queried agents are colored by EVoC clustering of their embeddings.
An arrow path connects the centroids of past batches up to the current round.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.axes
import matplotlib.lines
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.database as database
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import (
    AgentRegistryType,
    AlignedResultList,
    DataAgent,
    EstimatorStatistics,
    ScoredDataAgent,
)
from qcp_adaptive_termination.models.execution import AdditionalInformation, EvaluatorDecision
from qcp_adaptive_termination.models.query_plan import (
    MultiAgentQueryPlanStep,
    QueryPlan,
)
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.stages import execute_query_plan
from scripts.utils import (
    CATEGORICAL_COLORS,
    DB_GEO,
    NOISE_COLOR,
    ScriptContext,
    build_color_map,
    build_evoc_layout,
    run_script,
)

_SVG_PATH = Path(__file__).parent / "res" / "Continents.svg"

_BASE_SIZE: float = 60.0
_MAX_SIZE: float = 3.0 * _BASE_SIZE
_GRAY: str = "#aaaaaa"
_ALREADY_QUERIED_ALPHA: float = 0.55


# ── data structures ───────────────────────────────────────────────────────────


@dataclass(slots=True)
class _BatchRecord:
    step_id: str
    iteration: int
    batch: list[ScoredDataAgent]
    remaining: list[ScoredDataAgent]
    queried_before: list[str]
    result_list_before: AlignedResultList
    result_list_after: AlignedResultList | None
    decision: EvaluatorDecision | Literal["need"] | None


# ── hook registration ─────────────────────────────────────────────────────────


def _register_batch_hooks() -> list[_BatchRecord]:
    records: list[_BatchRecord] = []

    def _on_batch_start(
        execution_id: str,
        query_plan_step: MultiAgentQueryPlanStep,
        additional_information: list[AdditionalInformation],
        iteration: int,
        batch: list[ScoredDataAgent],
        queried: list[str],
        remaining: list[ScoredDataAgent],
        result_list: AlignedResultList,
    ) -> None:
        records.append(
            _BatchRecord(
                step_id=query_plan_step.id,
                iteration=iteration,
                batch=list(batch),
                remaining=list(remaining),
                queried_before=list(queried),
                result_list_before=result_list,
                result_list_after=None,
                decision=None,
            )
        )

    def _on_batch_end(
        execution_id: str,
        query_plan_step: MultiAgentQueryPlanStep,
        additional_information: list[AdditionalInformation],
        iteration: int,
        batch: list[ScoredDataAgent],
        remaining: list[ScoredDataAgent],
        result_list: AlignedResultList,
        estimators: list[EstimatorStatistics | None],
        decision: EvaluatorDecision | Literal["need"],
    ) -> None:
        for rec in reversed(records):
            if rec.step_id == query_plan_step.id and rec.iteration == iteration:
                rec.result_list_after = result_list
                rec.decision = decision
                break

    hooks.register(HookEvent.MULTI_AGENT_STEP_BATCH_START, _on_batch_start)
    hooks.register(HookEvent.MULTI_AGENT_STEP_BATCH_END, _on_batch_end)
    return records


# ── SQL helpers ───────────────────────────────────────────────────────────────


def _to_select_all_sql(sql: str) -> str:
    return re.sub(r"(?is)\bSELECT\b.*?\bFROM\b", "SELECT * FROM", sql, count=1)


def _run_select_all(agent: DataAgent, sql: str) -> pd.DataFrame:
    try:
        engine = database.get_database_engine(agent)
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn)
    except Exception:
        return pd.DataFrame()


# ── ground truth ──────────────────────────────────────────────────────────────


def _new_row_count(agent: DataAgent, select_all_sql: str, result_list: AlignedResultList) -> int:
    """Rows this agent would return that are not already collected."""
    agent_df = _run_select_all(agent, select_all_sql)
    if agent_df.empty:
        return 0

    if result_list.data.empty:
        return len(agent_df)

    source_col = result_list.schema_metadata.source_agent_column
    existing = result_list.data.drop(columns=[source_col], errors="ignore")

    existing_tuples: set[tuple[Any, ...]] = set(map(tuple, existing.values.tolist()))
    new_rows = sum(1 for row in agent_df.values.tolist() if tuple(row) not in existing_tuples)
    return max(0, new_rows)


def _compute_ground_truth(
    records: list[_BatchRecord],
    agents_by_name: dict[str, DataAgent],
    step_agent_names: list[str],
    select_all_sql: str,
) -> dict[tuple[str, int], dict[str, int]]:
    """Return (step_id, iteration) -> {agent_name: new_row_count}."""
    result: dict[tuple[str, int], dict[str, int]] = {}
    for rec in records:
        if rec.step_id not in {r.step_id for r in records if r.step_id == rec.step_id}:
            continue
        key = (rec.step_id, rec.iteration)
        if key in result:
            continue
        counts: dict[str, int] = {}
        for name in step_agent_names:
            agent = agents_by_name.get(name)
            if agent is None:
                continue
            counts[name] = _new_row_count(agent, select_all_sql, rec.result_list_before)
        result[key] = counts
    return result


# ── geo helpers ───────────────────────────────────────────────────────────────


def _agent_geo(agent: DataAgent) -> tuple[float, float] | None:
    stem = Path(agent.path).stem
    return DB_GEO.get(stem)


# ── EVoC colors ───────────────────────────────────────────────────────────────


async def _evoc_colors_for_step(
    step_agents: list[DataAgent],
    registry_type: AgentRegistryType,
) -> dict[str, str]:
    """agent_name -> hex color from EVoC clustering."""
    all_embeddings = await agent_registry.embedding_vectors_for_all_tables(step_agents, registry_type)
    names: list[str] = []
    vectors: list[list[float]] = []
    for agent in step_agents:
        tables = all_embeddings.get(agent.name, {})
        if not tables:
            continue
        mat = np.array(list(tables.values()), dtype=float)
        names.append(agent.name)
        vectors.append(mat.mean(axis=0).tolist())

    if len(vectors) < 2:
        default = CATEGORICAL_COLORS[0]
        return {n: default for n in names}

    X = np.array(vectors, dtype=float)
    _, clusterer = build_evoc_layout(X)
    top_labels = cast(np.ndarray, clusterer.labels_)
    color_map = build_color_map(clusterer.cluster_layers_, top_labels)
    fine_labels = clusterer.cluster_layers_[0] if len(clusterer.cluster_layers_) > 1 else top_labels

    colors: dict[str, str] = {}
    for i, name in enumerate(names):
        lbl = int(fine_labels[i])
        colors[name] = color_map.get(lbl, NOISE_COLOR) if lbl >= 0 else NOISE_COLOR
    return colors


# ── size normalization ────────────────────────────────────────────────────────


def _normalize_sizes(values: list[float]) -> list[float]:
    """Map values to [_BASE_SIZE, _MAX_SIZE]. Constant if no variance."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx == mn:
        return [(_BASE_SIZE + _MAX_SIZE) / 2.0] * len(values)
    return [_BASE_SIZE + (v - mn) / (mx - mn) * (_MAX_SIZE - _BASE_SIZE) for v in values]


# ── SVG background (adapted from embedding_visualization.py) ──────────────────


def _svg_d_to_mpl_path(d: str) -> matplotlib.path.Path | None:
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

    def _next(n: int) -> list[float]:
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
            x, y = _next(2)
            if cmd == "m":
                x, y = cx + x, cy + y
            cx, cy = x, y
            vertices.append((cx, cy))
            codes.append(MOVETO)
            cmd = "L" if cmd == "M" else "l"
        elif cmd in ("L", "l"):
            x, y = _next(2)
            if cmd == "l":
                x, y = cx + x, cy + y
            cx, cy = x, y
            vertices.append((cx, cy))
            codes.append(LINETO)
        elif cmd in ("H", "h"):
            (x,) = _next(1)
            if cmd == "h":
                x = cx + x
            cx = x
            vertices.append((cx, cy))
            codes.append(LINETO)
        elif cmd in ("V", "v"):
            (y,) = _next(1)
            if cmd == "v":
                y = cy + y
            cy = y
            vertices.append((cx, cy))
            codes.append(LINETO)
        elif cmd in ("C", "c"):
            coords = _next(6)
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
            coords = _next(4)
            if cmd == "s":
                coords = [cx + coords[0], cy + coords[1], cx + coords[2], cy + coords[3]]
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
            i += 1
            continue

    if len(vertices) < 2:
        return None
    return matplotlib.path.Path(np.array(vertices), np.array(codes, dtype=np.uint8))


def _parse_svg_paths() -> list[matplotlib.path.Path]:
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


_SVG_PATHS: list[matplotlib.path.Path] | None = None

_SVG_MX0, _SVG_MX1 = -109.73, 2651.67
_SVG_MY0, _SVG_MY1 = 40.94, 1484.71


def _draw_svg_background(ax: matplotlib.axes.Axes) -> None:
    global _SVG_PATHS
    if _SVG_PATHS is None:
        _SVG_PATHS = _parse_svg_paths()

    ax.set_facecolor("#ffffff")

    def _remap(verts: np.ndarray) -> np.ndarray:
        out = verts.copy()
        out[:, 0] = -180.0 + (verts[:, 0] - _SVG_MX0) / (_SVG_MX1 - _SVG_MX0) * 360.0
        out[:, 1] = 90.0 - (verts[:, 1] - _SVG_MY0) / (_SVG_MY1 - _SVG_MY0) * 180.0
        return out

    for svg_path in _SVG_PATHS:
        remapped = _remap(np.asarray(svg_path.vertices))
        remapped_path = matplotlib.path.Path(remapped, svg_path.codes)
        patch = matplotlib.patches.PathPatch(
            remapped_path,
            facecolor="#b9b9b9",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.7,
            zorder=1,
        )
        ax.add_patch(patch)


# ── plot rendering ────────────────────────────────────────────────────────────


def _render_selection_plot(
    step_id: str,
    iteration: int,
    variant: Literal["a", "b"],
    all_step_agents: list[DataAgent],
    queried_before_names: set[str],
    current_batch_agents: list[DataAgent],
    prior_agent_geos: list[tuple[float, float]],
    agent_sizes: dict[str, float],
    evoc_colors: dict[str, str],
    plot_path: Path,
    variant_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    _draw_svg_background(ax)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude", fontsize=9, color="#52514e")
    ax.set_ylabel("Latitude", fontsize=9, color="#52514e")

    current_names = {a.name for a in current_batch_agents}

    # ── scatter points ────────────────────────────────────────────────────────
    legend_handles: dict[str, matplotlib.lines.Line2D] = {}

    for agent in all_step_agents:
        geo = _agent_geo(agent)
        if geo is None:
            continue
        lon, lat = geo
        name = agent.name

        if name in queried_before_names:
            ax.scatter(
                lon,
                lat,
                c=_GRAY,
                s=_BASE_SIZE,
                linewidths=0.4,
                edgecolors="white",
                alpha=_ALREADY_QUERIED_ALPHA,
                zorder=3,
            )
        else:
            size = agent_sizes.get(name, (_BASE_SIZE + _MAX_SIZE) / 2.0)
            color = evoc_colors.get(name, CATEGORICAL_COLORS[0])
            zorder = 5 if name in current_names else 4
            linewidth = 1.2 if name in current_names else 0.4
            ax.scatter(
                lon,
                lat,
                c=color,
                s=size,
                linewidths=linewidth,
                edgecolors="white" if name not in current_names else "#222222",
                zorder=zorder,
            )
            if color not in legend_handles:
                legend_handles[color] = matplotlib.lines.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=7,
                    label=color,
                )

        ax.annotate(
            name,
            (lon, lat),
            fontsize=4.5,
            color="#333333",
            xytext=(3, 3),
            textcoords="offset points",
            clip_on=True,
            zorder=6,
        )

    # ── arrow path ────────────────────────────────────────────────────────────
    arrow_points: list[tuple[float, float]] = list(prior_agent_geos)
    for agent in current_batch_agents:
        geo = _agent_geo(agent)
        if geo is not None:
            arrow_points.append(geo)

    for idx in range(len(arrow_points) - 1):
        x0, y0 = arrow_points[idx]
        x1, y1 = arrow_points[idx + 1]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#333333",
                lw=1.3,
                mutation_scale=14,
                shrinkA=6,
                shrinkB=6,
            ),
            zorder=7,
        )

    # ── legend ────────────────────────────────────────────────────────────────
    extra_handles = [
        matplotlib.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_GRAY,
            markersize=6,
            alpha=_ALREADY_QUERIED_ALPHA,
            label="Already queried",
        ),
    ]
    all_handles = extra_handles + list(legend_handles.values())
    ax.legend(handles=all_handles, loc="lower left", fontsize=6.5, framealpha=0.85, edgecolor="#cccccc")

    title = (
        f"Step {step_id} · Round {iteration + 1}"
        f" · Variant {'a (model score)' if variant == 'a' else 'b (ground-truth count)'}"
        f"\n{variant_label}"
    )
    ax.set_title(title, fontsize=10, color="#0b0b0b")
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e0e0e0")

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_path.name}")


# ── per-step plot generation ──────────────────────────────────────────────────


def _generate_step_plots(
    step_id: str,
    step_records: list[_BatchRecord],
    all_step_agents: list[DataAgent],
    evoc_colors: dict[str, str],
    ground_truth: dict[tuple[str, int], dict[str, int]],
    output_dir: Path,
) -> None:
    agents_by_name = {a.name: a for a in all_step_agents}

    sorted_records = sorted(step_records, key=lambda r: r.iteration)

    for rec in sorted_records:
        # Ordered geos of all agents queried in previous rounds (batch order within each round).
        past_agent_geos: list[tuple[float, float]] = [
            geo
            for r in sorted_records
            if r.iteration < rec.iteration
            for sda in r.batch
            if sda.agent.name in agents_by_name and (geo := _agent_geo(agents_by_name[sda.agent.name])) is not None
        ]
        queried_before_set_base = set(rec.queried_before)
        to_query_agents = [a for a in all_step_agents if a.name not in queried_before_set_base]
        to_query_names = [a.name for a in to_query_agents]

        scored_map: dict[str, float] = {}
        for sda in rec.batch + rec.remaining:
            if sda.score is not None:
                scored_map[sda.agent.name] = sda.score

        raw_scores_a = [scored_map.get(n, 0.0) for n in to_query_names]
        sizes_a_list = _normalize_sizes(raw_scores_a)
        sizes_a = dict(zip(to_query_names, sizes_a_list, strict=True))
        score_min = min(raw_scores_a, default=0.0)
        score_max = max(raw_scores_a, default=0.0)

        gt = ground_truth.get((step_id, rec.iteration), {})
        raw_counts_b = [float(gt.get(n, 0)) for n in to_query_names]
        sizes_b_list = _normalize_sizes(raw_counts_b)
        sizes_b = dict(zip(to_query_names, sizes_b_list, strict=True))
        count_min = min(raw_counts_b, default=0.0)
        count_max = max(raw_counts_b, default=0.0)

        batch_agents = [agents_by_name[sda.agent.name] for sda in rec.batch if sda.agent.name in agents_by_name]

        for j, current_agent in enumerate(batch_agents):
            # Geos of agents already queried in this batch (selection order).
            intra_geos: list[tuple[float, float]] = [
                geo for a in batch_agents[:j] if (geo := _agent_geo(a)) is not None
            ]
            prior_agent_geos = past_agent_geos + intra_geos
            queried_before_names = queried_before_set_base | {a.name for a in batch_agents[:j]}

            label_a = f"[{j + 1}/{len(batch_agents)}] Score range: [{score_min:.3f}, {score_max:.3f}]"
            _render_selection_plot(
                step_id=step_id,
                iteration=rec.iteration,
                variant="a",
                all_step_agents=all_step_agents,
                queried_before_names=queried_before_names,
                current_batch_agents=[current_agent],
                prior_agent_geos=prior_agent_geos,
                agent_sizes=sizes_a,
                evoc_colors=evoc_colors,
                plot_path=output_dir / f"step_{step_id}_variant_a_round_{rec.iteration:02d}_agent_{j:02d}.png",
                variant_label=label_a,
            )

            label_b = f"[{j + 1}/{len(batch_agents)}] New-row count range: [{int(count_min)}, {int(count_max)}]"
            _render_selection_plot(
                step_id=step_id,
                iteration=rec.iteration,
                variant="b",
                all_step_agents=all_step_agents,
                queried_before_names=queried_before_names,
                current_batch_agents=[current_agent],
                prior_agent_geos=prior_agent_geos,
                agent_sizes=sizes_b,
                evoc_colors=evoc_colors,
                plot_path=output_dir / f"step_{step_id}_variant_b_round_{rec.iteration:02d}_agent_{j:02d}.png",
                variant_label=label_b,
            )


# ── logging ───────────────────────────────────────────────────────────────────


def _log_batch_records(records: list[_BatchRecord]) -> None:
    print(f"\n=== Selection path — {len(records)} batch round(s) captured ===")
    for rec in records:
        batch_names = [s.agent.name for s in rec.batch]
        batch_scores = [f"{s.score:.4f}" if s.score is not None else "None" for s in rec.batch]
        print(
            f"  Step {rec.step_id} round {rec.iteration}: "
            f"batch={batch_names} scores={batch_scores} "
            f"queried_before={rec.queried_before} "
            f"remaining={len(rec.remaining)} decision={rec.decision}"
        )
    print()


# ── main ──────────────────────────────────────────────────────────────────────


async def _run(ctx: ScriptContext) -> None:
    extra = ctx.extra or {}
    query_plan_dict: dict = extra.get("query_plan", {})
    sql_query: str = extra.get("sql_query", "")

    if not query_plan_dict:
        raise ValueError("extra.query_plan is missing from input file")
    if not sql_query:
        raise ValueError("extra.sql_query is missing from input file")

    query_plan = QueryPlan.model_validate(query_plan_dict)
    select_all_sql = _to_select_all_sql(sql_query)

    # Extract agents per step directly from the query plan.
    multi_steps: list[MultiAgentQueryPlanStep] = [
        step for step in query_plan.steps if isinstance(step, MultiAgentQueryPlanStep)
    ]
    if not multi_steps:
        print("No MultiAgentQueryPlanStep found in query plan — nothing to visualize.")
        return

    agents_by_name = {a.name: a for a in ctx.data_agents}
    step_agents: dict[str, list[DataAgent]] = {}
    for step in multi_steps:
        step_agents[step.id] = [agents_by_name[n] for n in step.to_agent if n in agents_by_name]

    relevant_data_agents: list[DataAgent] = list(
        {a.name: a for agents in step_agents.values() for a in agents}.values()
    )

    print(
        f"Running stage 3 for {len(multi_steps)} multi-agent step(s) with {len(relevant_data_agents)} relevant agents …"
    )

    batch_records = _register_batch_hooks()

    try:
        await execute_query_plan(
            query_plan=query_plan,
            relevant_data_agents=relevant_data_agents,
            data_agents=ctx.data_agents,
        )
    finally:
        hooks.clear(HookEvent.MULTI_AGENT_STEP_BATCH_START)
        hooks.clear(HookEvent.MULTI_AGENT_STEP_BATCH_END)

    _log_batch_records(batch_records)

    if not batch_records:
        print("No batch records collected — no plots to generate.")
        return

    output_dir = context.get_output_dir()
    assert output_dir is not None, "output_dir must be set (pass --output_dir)"

    registry_type = AgentRegistryType(ctx.config.selection_kwargs.get("registry_type", AgentRegistryType.TECHNICAL))

    # Process each step independently.
    for step in multi_steps:
        sid = step.id
        records_for_step = [r for r in batch_records if r.step_id == sid]
        if not records_for_step:
            print(f"Step {sid}: no batch records collected, skipping plots.")
            continue

        agents_this_step = step_agents[sid]
        agent_names_this_step = [a.name for a in agents_this_step]

        print(f"Step {sid}: computing ground truth for {len(agents_this_step)} agents …")
        gt = _compute_ground_truth(records_for_step, agents_by_name, agent_names_this_step, select_all_sql)

        print(f"Step {sid}: computing EVoC colors …")
        evoc_colors = await _evoc_colors_for_step(agents_this_step, registry_type)

        total_plots = sum(len(r.batch) for r in records_for_step) * 2
        print(f"Step {sid}: generating {total_plots} plot(s) …")
        _generate_step_plots(
            step_id=sid,
            step_records=records_for_step,
            all_step_agents=agents_this_step,
            evoc_colors=evoc_colors,
            ground_truth=gt,
            output_dir=output_dir,
        )

    print("Done.")


if __name__ == "__main__":
    run_script(
        description="Visualize adaptive selection path for multi-agent query plan execution",
        run_fn=_run,
    )
