from enum import Enum
import os
from typing import cast
from dataclasses import dataclass
from sqlalchemy import Inspector, create_engine, inspect, text
import random
import datetime
import uuid

from sqlalchemy.engine import Engine
from sqlalchemy import event

from qtypes.data_agent import DataAgent


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA temp_store=MEMORY;")
    cursor.execute("PRAGMA foreign_keys=OFF;")
    cursor.execute("PRAGMA synchronous=OFF;")
    cursor.close()


class SplitType(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    MIXED = "mixed"


class NoiseType(Enum):
    NONE = "none"
    # TRANSLATE, RENAME_COLUMNS, CHANGE_DATA_TYPE, USE_DATA_ABBREVIATIONS, SYNONYMS


@dataclass
class SplitTree:
    table_name: str  # the table this split belongs to
    columns: list[
        str
    ]  # all columns in the table that are in this split (including always included columns)
    always_included_columns: list[
        str
    ]  # e.g. primary key columns that always need to be included in the split to join with other splits of the same table
    rows: tuple[int, int]  # (start_row inclusive start at 0, end_row exclusive)
    children: list["SplitTree"]

    def leaf_nodes(self) -> list["SplitTree"]:
        if not self.children:
            return [self]

        leaves = []
        for child in self.children:
            leaves.extend(child.leaf_nodes())
        return leaves

    def amount_splits_possible(self, split_type: SplitType) -> int:
        if split_type == SplitType.HORIZONTAL:
            return self.rows[1] - self.rows[0]
        elif split_type == SplitType.VERTICAL:
            return len(self.columns) - len(self.always_included_columns)
        elif split_type == SplitType.MIXED:
            horizontal_splits = self.rows[1] - self.rows[0]
            vertical_splits = len(self.columns) - len(self.always_included_columns)
            return horizontal_splits * vertical_splits
        else:
            raise ValueError(f"Invalid split type: {split_type}")

    def split_n_times_horizontal(self, n: int, seed: int = 42) -> bool:
        random.seed(seed)

        if n == 1:
            return True

        try:
            split_points = sorted(
                random.sample(range(self.rows[0] + 1, self.rows[1]), n - 1)
            )
            split_points = [self.rows[0]] + split_points + [self.rows[1]]
        except ValueError:
            print(f"Not enough rows to split horizontally for rows {self.rows}")
            return False

        self.children = []
        for i in range(n):
            child = SplitTree(
                table_name=self.table_name,
                columns=self.columns,
                always_included_columns=self.always_included_columns,
                rows=(split_points[i], split_points[i + 1]),
                children=[],
            )
            self.children.append(child)
        return True

    def split_n_times_vertical(self, n: int, seed: int = 42) -> bool:
        random.seed(seed)

        if n == 1:
            return True

        # exclude always_included_columns from the split
        try:
            split_columns = [
                col for col in self.columns if col not in self.always_included_columns
            ]
            split_points = sorted(random.sample(range(1, len(split_columns)), n - 1))
            split_points = [0] + split_points + [len(split_columns)]
        except ValueError:
            print(
                f"Not enough columns to split vertically for columns {self.columns} and always included columns {self.always_included_columns}"
            )
            return False

        self.children = []
        for i in range(n):
            child_columns = (
                self.always_included_columns
                + split_columns[split_points[i] : split_points[i + 1]]
            )
            child = SplitTree(
                table_name=self.table_name,
                columns=child_columns,
                always_included_columns=self.always_included_columns,
                rows=self.rows,
                children=[],
            )
            self.children.append(child)
        return True

    def split_n_times_mixed(self, n: int, seed: int = 42) -> bool:
        random.seed(seed)

        if n == 1:
            return True

        # Keep semantics aligned with horizontal/vertical splitters:
        # n means resulting number of leaf splits, so we need n-1 split operations.
        target_split_operations = n - 1
        nodes_splitted = 0
        last_split_was_horizontal = random.random() < 0.5

        nodes_to_split: list[SplitTree] = [self]
        while nodes_splitted < target_split_operations:
            if len(nodes_to_split) == 0:
                print(f"Not enough splits possible for mixed split with n={n}")
                return False

            node = random.choice(nodes_to_split)
            nodes_to_split.remove(node)

            is_vertical_split_possible = (
                len(node.columns) - len(node.always_included_columns) >= 2
            )
            is_horizontal_split_possible = node.rows[1] - node.rows[0] >= 2

            if not is_vertical_split_possible and not is_horizontal_split_possible:
                continue

            recursive_seed = random.randint(0, 1000)
            preferred = (
                ["vertical", "horizontal"]
                if last_split_was_horizontal
                else ["horizontal", "vertical"]
            )
            if not is_vertical_split_possible:
                preferred.remove("vertical")
            if not is_horizontal_split_possible:
                preferred.remove("horizontal")

            splitted = False
            for split_type in preferred:
                if split_type == "vertical":
                    splitted = node.split_n_times_vertical(2, recursive_seed)
                    if splitted:
                        last_split_was_horizontal = False
                        break
                else:
                    splitted = node.split_n_times_horizontal(2, recursive_seed)
                    if splitted:
                        last_split_was_horizontal = True
                        break

            if not splitted:
                continue

            nodes_splitted += 1
            nodes_to_split.extend(node.children)

        return True


@dataclass
class AgentConfig:
    agent_name: str
    db_file: str
    splits: list[SplitTree]  # list of splits assigned to the agent


def create_split_tree_for_table(
    table: str,
    input_db_engine: Engine,
    input_db_engine_inspector: Inspector,
    always_included_columns_mapping: dict[str, list[str]],
) -> SplitTree:
    with input_db_engine.connect() as conn:
        row_count = cast(
            int, conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
        )
        print(f"Table {table}: {row_count} rows")

    always_included_columns = input_db_engine_inspector.get_pk_constraint(table)[
        "constrained_columns"
    ]
    if len(always_included_columns) == 0:
        always_included_columns = always_included_columns_mapping.get(table, [])

    if len(always_included_columns) == 0:
        fk_cols = list(
            {col for fk in input_db_engine_inspector.get_foreign_keys(table) for col in fk["constrained_columns"]}
        )
        always_included_columns = fk_cols

    if len(always_included_columns) == 0:
        raise Exception(
            f"No primary key found for table {table} and no mapping available, cannot proceed with splitting"
        )

    return SplitTree(
        table_name=table,
        columns=[col["name"] for col in input_db_engine_inspector.get_columns(table)],
        always_included_columns=always_included_columns,
        rows=(0, row_count),
        children=[],
    )


def get_splits_for_table(
    table: str,
    input_db_engine: Engine,
    input_db_engine_inspector: Inspector,
    always_included_columns_mapping: dict[str, list[str]],
    split_type: SplitType,
    force_splits: bool,
    max_splits_per_table: int | None,
    seed: int,
    agent_amount: int,
) -> tuple[SplitTree, list[SplitTree]]:
    print(
        f"================================================== CREATING {split_type.name} SPLITS FOR {table.upper()} =================================================="
    )
    split_tree = create_split_tree_for_table(
        table,
        input_db_engine,
        input_db_engine_inspector,
        always_included_columns_mapping,
    )

    min_splits = 1 + int(force_splits)
    max_splits = min(split_tree.amount_splits_possible(split_type) + 1, agent_amount)
    if max_splits_per_table is not None:
        max_splits = min(max_splits, max_splits_per_table)

    chosen_splits = random.randint(min_splits, max_splits)

    success = False
    while not success and chosen_splits > 1:
        if split_type == SplitType.HORIZONTAL:
            success = split_tree.split_n_times_horizontal(chosen_splits, seed)
        elif split_type == SplitType.VERTICAL:
            success = split_tree.split_n_times_vertical(chosen_splits, seed)
        elif split_type == SplitType.MIXED:
            success = split_tree.split_n_times_mixed(chosen_splits, seed)
        else:
            raise ValueError(f"Invalid split type: {split_type}")

        chosen_splits -= 1

    splits = split_tree.leaf_nodes()

    print(
        f"Table {table}: {len(splits)} (min: {min_splits}, max: {max_splits}) {split_type.value} splits"
    )
    for split in splits:
        print(f"    {split}")

    return split_tree, splits


def create_splits_visualization(
    table_splits_mapping: dict[str, SplitTree],
    data_agents_config: list[AgentConfig],
    output_file: str,
):
    print(
        "================================================== SPLIT ASSIGNMENTS TO AGENTS =================================================="
    )
    for data_agent_config in data_agents_config:
        print(f"Agent {data_agent_config.agent_name} assigned splits:")
        for split in data_agent_config.splits:
            print(f"    {split}")

    # AI GENERATED
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    def escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def wrap_text(text: str, max_chars: int) -> list[str]:
        words = text.split()
        if not words:
            return [""]

        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def split_key(split: SplitTree) -> tuple[str, tuple[int, int], tuple[str, ...]]:
        return (split.table_name, split.rows, tuple(split.columns))

    agent_colors = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]
    agent_color_map = {
        config.agent_name: agent_colors[index % len(agent_colors)]
        for index, config in enumerate(data_agents_config)
    }

    split_to_agent: dict[tuple[str, tuple[int, int], tuple[str, ...]], str] = {}
    for agent_config in data_agents_config:
        for split in agent_config.splits:
            split_to_agent[split_key(split)] = agent_config.agent_name

    def assigned_agent(split: SplitTree) -> str | None:
        return split_to_agent.get(split_key(split))

    def row_range_label(split: SplitTree) -> str:
        start_row, end_row = split.rows
        if end_row - start_row == 0:
            return "rows: empty"
        return f"rows: {start_row}..{end_row - 1} ({end_row - start_row})"

    def column_label(split: SplitTree) -> str:
        if not split.columns:
            return "cols: none"
        return f"cols: {', '.join(split.columns)}"

    def render_text_block(
        x: int,
        y: int,
        lines: list[str],
        font_size: int,
        fill: str = "#1F2937",
        weight: str = "normal",
    ) -> list[str]:
        rendered: list[str] = []
        for index, line in enumerate(lines):
            dy = y + index * int(font_size * 1.25)
            rendered.append(
                f'<text x="{x}" y="{dy}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="{font_size}" '
                f'font-weight="{weight}" fill="{fill}">{escape(line)}</text>'
            )
        return rendered

    def render_card(
        x: int,
        y: int,
        width: int,
        title: str,
        subtitle: str,
        items: list[SplitTree],
        kind: str,
    ) -> tuple[list[str], int]:
        title_font = 24
        subtitle_font = 13
        item_title_font = 15
        item_body_font = 12
        left_padding = 18
        right_padding = 18
        top_padding = 18
        bottom_padding = 18
        item_gap = 12

        cols = 2 if width >= 900 else 1
        col_gap = 14
        tile_width = (
            width - left_padding - right_padding - col_gap * (cols - 1)
        ) // cols

        def tile_text_lines(split: SplitTree) -> tuple[list[str], list[str]]:
            title_lines = [split.table_name, row_range_label(split)]
            agent_name = assigned_agent(split)
            if agent_name is not None:
                title_lines.append(f"assigned to {agent_name}")
            detail_lines = wrap_text(column_label(split), 58)
            return title_lines, detail_lines

        def tile_height(split: SplitTree) -> int:
            title_lines, detail_lines = tile_text_lines(split)
            title_height = len(title_lines) * 22
            detail_height = len(detail_lines) * 16
            base_height = 28 + title_height + 10 + detail_height + 20
            return max(118 if kind == "table" else 96, base_height)

        row_heights: list[int] = []
        for row_start in range(0, len(items), cols):
            row_items = items[row_start : row_start + cols]
            row_heights.append(max(tile_height(split) for split in row_items))

        body_height = sum(row_heights) + max(0, len(row_heights) - 1) * item_gap
        total_height = top_padding + 38 + 26 + body_height + bottom_padding

        elements: list[str] = []
        elements.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{total_height}" rx="18" ry="18" fill="#FFFFFF" stroke="#D6DAE3" stroke-width="1.5"/>'
        )
        elements.extend(
            render_text_block(
                x + left_padding,
                y + 30,
                [title],
                title_font,
                fill="#111827",
                weight="700",
            )
        )
        elements.extend(
            render_text_block(
                x + left_padding,
                y + 52,
                wrap_text(subtitle, 90),
                subtitle_font,
                fill="#6B7280",
            )
        )

        body_top = y + top_padding + 60
        row_y = body_top
        for row_index, row_start in enumerate(range(0, len(items), cols)):
            row_items = items[row_start : row_start + cols]
            row_height = row_heights[row_index]
            for col, split in enumerate(row_items):
                tile_x = x + left_padding + col * (tile_width + col_gap)
                tile_y = row_y
                agent_name = assigned_agent(split)
                accent = (
                    agent_color_map[agent_name]
                    if agent_name is not None and agent_name in agent_color_map
                    else "#CBD5E1"
                )
                fill = "#F8FAFC" if kind == "table" else "#F9FAFB"

                current_item_height = tile_height(split)
                elements.append(
                    f'<rect x="{tile_x}" y="{tile_y}" width="{tile_width}" height="{current_item_height}" rx="14" ry="14" fill="{fill}" stroke="{accent}" stroke-width="2"/>'
                )
                elements.append(
                    f'<rect x="{tile_x}" y="{tile_y}" width="8" height="{current_item_height}" rx="4" ry="4" fill="{accent}"/>'
                )

                title_lines, detail_lines = tile_text_lines(split)

                text_x = tile_x + 18
                text_y = tile_y + 24
                elements.extend(
                    render_text_block(
                        text_x,
                        text_y,
                        title_lines,
                        item_title_font,
                        fill="#111827",
                        weight="600",
                    )
                )

                detail_y = text_y + len(title_lines) * 22 + 8
                elements.extend(
                    render_text_block(
                        text_x, detail_y, detail_lines, item_body_font, fill="#4B5563"
                    )
                )

            row_y += row_height + item_gap

        return elements, total_height

    def render_agent_chip(
        x: int, y: int, width: int, split: SplitTree
    ) -> tuple[list[str], int]:
        agent_name = assigned_agent(split)
        color = (
            agent_color_map[agent_name]
            if agent_name is not None and agent_name in agent_color_map
            else "#CBD5E1"
        )
        detail_lines = wrap_text(
            f"{row_range_label(split)} | {column_label(split)}", 66
        )
        chip_height = max(92, 42 + len(detail_lines) * 18)
        elements: list[str] = []
        elements.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{chip_height}" rx="12" ry="12" fill="#FFFFFF" stroke="{color}" stroke-width="2"/>'
        )
        elements.append(
            f'<rect x="{x}" y="{y}" width="10" height="{chip_height}" rx="12" ry="12" fill="{color}"/>'
        )
        title = f"{split.table_name} • {row_range_label(split)}"
        if agent_name is not None:
            detail_lines = [agent_name] + detail_lines
        elements.extend(
            render_text_block(
                x + 18,
                y + 22,
                wrap_text(title, 60)[:2],
                13,
                fill="#111827",
                weight="600",
            )
        )
        elements.extend(
            render_text_block(x + 18, y + 44, detail_lines, 11, fill="#4B5563")
        )
        return elements, chip_height

    table_order = list(table_splits_mapping.items())
    section_gap = 28
    outer_margin = 26
    header_height = 84
    left_x = outer_margin
    table_width = 980
    agent_width = 420
    right_x = left_x + table_width + section_gap
    canvas_width = left_x + table_width + section_gap + agent_width + outer_margin

    table_elements: list[str] = []
    agent_elements: list[str] = []

    current_y = outer_margin + header_height
    table_elements.append(
        f'<text x="{left_x}" y="{outer_margin + 34}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="30" font-weight="700" fill="#111827">Split overview</text>'
    )
    table_elements.append(
        f'<text x="{left_x}" y="{outer_margin + 58}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="14" fill="#6B7280">Tables on the left, agent assignments on the right.</text>'
    )

    for table_name, split_tree in table_order:
        leaf_splits = split_tree.leaf_nodes()
        card_elements, card_height = render_card(
            left_x,
            current_y,
            table_width,
            f"Table {table_name}",
            f"{len(leaf_splits)} split{'s' if len(leaf_splits) != 1 else ''} produced from {split_tree.rows[1] - split_tree.rows[0]} rows and {len(split_tree.columns)} columns",
            leaf_splits,
            kind="table",
        )
        table_elements.extend(card_elements)
        current_y += card_height + 18

    table_section_height = current_y + outer_margin

    agent_elements.append(
        f'<text x="{right_x}" y="{outer_margin + 34}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="30" font-weight="700" fill="#111827">Agent assignments</text>'
    )
    agent_elements.append(
        f'<text x="{right_x}" y="{outer_margin + 58}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="14" fill="#6B7280">Each agent gets a color; matching table splits use the same accent.</text>'
    )

    current_agent_y = outer_margin + header_height
    for agent_index, agent_config in enumerate(data_agents_config):
        agent_color = agent_color_map.get(
            agent_config.agent_name, agent_colors[agent_index % len(agent_colors)]
        )
        agent_splits = agent_config.splits
        card_height = 108 + max(0, len(agent_splits) - 1) * 104
        agent_elements.append(
            f'<rect x="{right_x}" y="{current_agent_y}" width="{agent_width}" height="{card_height}" rx="18" ry="18" fill="#FFFFFF" stroke="#D6DAE3" stroke-width="1.5"/>'
        )
        agent_elements.append(
            f'<rect x="{right_x}" y="{current_agent_y}" width="10" height="{card_height}" rx="18" ry="18" fill="{agent_color}"/>'
        )
        agent_elements.extend(
            render_text_block(
                right_x + 18,
                current_agent_y + 26,
                [agent_config.agent_name],
                22,
                fill="#111827",
                weight="700",
            )
        )
        agent_elements.extend(
            render_text_block(
                right_x + 18,
                current_agent_y + 48,
                [
                    f"{len(agent_splits)} assigned split{'s' if len(agent_splits) != 1 else ''}"
                ],
                13,
                fill="#6B7280",
            )
        )

        chip_y = current_agent_y + 68
        if agent_splits:
            for split in agent_splits:
                chip_elements, chip_height = render_agent_chip(
                    right_x + 14, chip_y, agent_width - 28, split
                )
                agent_elements.extend(chip_elements)
                chip_y += chip_height + 12
        else:
            agent_elements.extend(
                render_text_block(
                    right_x + 18,
                    chip_y + 18,
                    ["No splits assigned"],
                    12,
                    fill="#6B7280",
                )
            )

        current_agent_y = max(current_agent_y + card_height + 18, chip_y + 6)

    agent_section_height = current_agent_y + outer_margin
    canvas_height = max(table_section_height, agent_section_height)

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">',
        "<defs>",
        '<linearGradient id="backgroundGradient" x1="0%" y1="0%" x2="100%" y2="100%">',
        '<stop offset="0%" stop-color="#F8FAFC"/>',
        '<stop offset="100%" stop-color="#EEF2FF"/>',
        "</linearGradient>",
        "</defs>",
        f'<rect x="0" y="0" width="{canvas_width}" height="{canvas_height}" fill="url(#backgroundGradient)"/>',
        f'<rect x="{left_x - 10}" y="{outer_margin - 8}" width="{table_width + 20}" height="{canvas_height - 2 * outer_margin + 6}" rx="26" ry="26" fill="#FFFFFF" fill-opacity="0.28" stroke="#FFFFFF" stroke-opacity="0.35" stroke-width="1"/>',
        f'<rect x="{right_x - 10}" y="{outer_margin - 8}" width="{agent_width + 20}" height="{canvas_height - 2 * outer_margin + 6}" rx="26" ry="26" fill="#FFFFFF" fill-opacity="0.28" stroke="#FFFFFF" stroke-opacity="0.35" stroke-width="1"/>',
        *table_elements,
        *agent_elements,
        "</svg>",
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))

    print(f"Wrote split visualization to {output_file}")


def create_agents(
    output_path: str,
    database_url: str,
    agent_amount: int,
    always_included_columns_mapping: dict[str, list[str]],
    split_type: SplitType = SplitType.MIXED,
    noise_type: NoiseType = NoiseType.NONE,
    force_splits: bool = False,
    max_splits_per_table: int | None = None,
    seed: int = 42,
) -> list[DataAgent]:
    if agent_amount < 1:
        raise ValueError("agent_amount must be at least 1")

    random.seed(seed)

    os.makedirs(output_path, exist_ok=True)

    # write config file config.txt in output dir
    with open(f"{output_path}/config.txt", "w") as f:
        f.write(f"output_path: {output_path}\n")
        f.write(f"database_url: {database_url}\n")
        f.write(f"agent_amount: {agent_amount}\n")
        f.write(f"split_type: {split_type.value}\n")
        f.write(f"noise_type: {noise_type.value}\n")
        f.write(f"force_splits: {force_splits}\n")
        f.write(f"seed: {seed}\n")

    input_db_engine = create_engine(database_url)
    input_db_engine_inspector = inspect(input_db_engine)
    table_names = input_db_engine_inspector.get_table_names()

    data_agents_config: list[AgentConfig] = [
        AgentConfig(agent_name=f"A{i}", db_file=f"{output_path}/A{i}.db", splits=[])
        for i in range(1, agent_amount + 1)
    ]

    table_splits_mapping: dict[str, SplitTree] = {}

    # 1. for each table assign table to 1..min(splits_possible, agent_amount) agents based on split_type and seed
    # prefer to assign to agents that have less tables assigned already to balance the amount of tables per agent
    for table in table_names:
        recursive_seed = random.randint(0, 1000)
        root, splits = get_splits_for_table(
            table,
            input_db_engine,
            input_db_engine_inspector,
            always_included_columns_mapping,
            split_type,
            force_splits,
            max_splits_per_table,
            recursive_seed,
            agent_amount,
        )
        table_splits_mapping[table] = root

        # assign splits to agents, if there exists a table with 0 table_names chose that one (only if table names does not exist)
        for split in splits:
            # first assign to agents without splits
            split_assigned = False
            for agent_config in data_agents_config:
                if len(agent_config.splits) > 0:
                    continue  # skip agents that already have splits assigned
                agent_config.splits.append(split)
                split_assigned = True
                break

            if split_assigned:
                continue

            # don't assign two splits from same table to same agent
            possible_agents = [
                agent_config
                for agent_config in data_agents_config
                if table not in [s.table_name for s in agent_config.splits]
            ]
            # randomly chose one of the possible agents
            agent_config = random.choice(possible_agents)
            agent_config.splits.append(split)

    # 2. create info graphic
    create_splits_visualization(
        table_splits_mapping,
        data_agents_config,
        output_file=f"{output_path}/splits_visualization.svg",
    )

    # Fully close all pooled connections to the source DB before creating agent DBs
    input_db_engine.dispose()

    # 3. create new db's with tables for each agent
    print(
        "================================================== CREATING AGENT DATABASES =================================================="
    )
    for data_agent_config in data_agents_config:
        print(
            f"Creating database for agent {data_agent_config.agent_name} with splits:"
        )
        agent_db_file = data_agent_config.db_file
        if os.path.exists(agent_db_file):
            os.remove(agent_db_file)

        agent_engine = create_engine(f"sqlite:///{os.path.abspath(agent_db_file)}")

        # create table statements
        with agent_engine.connect() as conn:
            for split in data_agent_config.splits:
                columns = [
                    column
                    for column in input_db_engine_inspector.get_columns(
                        split.table_name
                    )
                    if column["name"] in split.columns
                ]
                columns_definition = ", ".join(
                    [f'"{col["name"]}" {col["type"]}' for col in columns]
                )
                create_table_query = (
                    f'CREATE TABLE "{split.table_name}" ({columns_definition})'
                )
                print("    " + create_table_query)
                conn.execute(text(create_table_query))

        agent_engine.dispose()

    # 4. fill data (multithreaded for each table)
    print(
        "================================================== FILLING AGENT DATABASES WITH DATA =================================================="
    )
    for agent_index, data_agent_config in enumerate(data_agents_config):
        attach_alias = f"target_{agent_index}"
        with input_db_engine.connect() as conn:
            conn.exec_driver_sql(
                f"ATTACH DATABASE '{os.path.abspath(data_agent_config.db_file)}' AS {attach_alias};"
            )
            conn.exec_driver_sql(f"PRAGMA {attach_alias}.journal_mode=WAL;")
            conn.exec_driver_sql(f"PRAGMA {attach_alias}.foreign_keys=OFF;")
            conn.exec_driver_sql(f"PRAGMA {attach_alias}.journal_mode=MEMORY;")
            conn.exec_driver_sql(f"PRAGMA {attach_alias}.temp_store=MEMORY;")
            conn.exec_driver_sql(f"PRAGMA {attach_alias}.synchronous=OFF;")

            # insert into target.(table_name) (only relevant columns for split)
            # select (only relevant columns for split)
            # from main.(table_name)
            # LIMIT (split.rows[1] - split.rows[0]) OFFSET split.rows[0]
            for split in data_agent_config.splits:
                columns_str = ", ".join(f'"{c}"' for c in split.columns)
                query = f"""
                    INSERT INTO {attach_alias}."{split.table_name}" ({columns_str})
                    SELECT {columns_str}
                    FROM main."{split.table_name}"
                    LIMIT {split.rows[1] - split.rows[0]} OFFSET {split.rows[0]}
                """
                print(f"    Inserting data for split {split} with query: {query}")
                conn.execute(text(query))

            conn.commit()  # SQLite requires no active write transaction when detaching a database.
            conn.exec_driver_sql(f"DETACH DATABASE {attach_alias};")
            conn.close()

    # 5. return data agents
    data_agents: list[DataAgent] = []
    for data_agent_config in data_agents_config:
        data_agent = DataAgent(
            type="sql",
            name=data_agent_config.agent_name,
            path=data_agent_config.db_file,
        )

        data_agents.append(data_agent)
        print(
            f"{data_agent.name}: 'sqlite:///{os.path.abspath(data_agent_config.db_file)}'"
        )

    return data_agents
