"""Render a run's raw conversation into a full, interleaved Markdown transcript.

The transcript is derived purely from the raw ``continuation_input`` (the
``to_input_list()`` dicts: reasoning, messages, tool calls/outputs), which is
also what gets cached. This keeps the cache bare metal and lets rendering change
independently of how runs are executed.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from agents.items import TResponseInputItem

from qcp_adaptive_termination.models import AgentRole


@dataclass(slots=True)
class RunMeta:
    """Identifiers attached to a run for naming + the transcript header."""

    role: AgentRole
    agent_name: str | None
    label: str | None


def _content_text(content: Any) -> str:
    """Flatten a message content list/string into text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("text"))
    return str(content or "")


def _item_blocks(item: TResponseInputItem) -> list[str]:
    """Render a single raw conversation item into Markdown lines."""
    if not isinstance(item, dict):
        return []
    i_type = item.get("type")
    if i_type == "reasoning":
        summaries = item.get("summary") or []
        text = "\n".join(s.get("text", "") for s in summaries if isinstance(s, dict) and s.get("text"))
        return ["### Reasoning", "", text, ""] if text else []
    if i_type == "message":
        role = item.get("role") or "message"
        text = _content_text(item.get("content")).strip()
        return [f"### {str(role).capitalize()}", "", text, ""] if text else []
    if i_type == "function_call":
        name = item.get("name") or ""
        args = str(item.get("arguments") or "")
        if name == "execute_code":
            try:
                code = json.loads(args).get("code")
                if isinstance(code, str):
                    args = code
            except Exception:
                pass
        return [f"### Tool call `{name}`", "", "```", args, "```", ""]
    if i_type == "function_call_output":
        return ["### Tool result", "", "```", str(item.get("output", "")).strip(), "```", ""]
    return []


def render_run_markdown(
    meta: RunMeta,
    input: list[TResponseInputItem],
    final_output: Any | None = None,
) -> str:
    """Render a run's conversation to a complete Markdown transcript."""
    lines = [f"# Agent run — {meta.role.value}", ""]
    if meta.agent_name:
        lines += [f"**Agent:** `{meta.agent_name}`", ""]
    if meta.label:
        lines += [f"**Label:** `{meta.label}`", ""]

    if input:
        lines += ["## Run", ""]
        for item in input:
            lines += _item_blocks(item)

    if final_output is not None:
        lines += ["## Output", "", "```", str(final_output), "```", ""]

    return "\n".join(lines).rstrip() + "\n"


def transcript_filename(meta: RunMeta) -> str:
    """Derive a meaningful filename from a run's identifiers.

    Uses the label when present (carries step + agents_queried, e.g.
    ``COMPLETENESS step=s1 agents_queried=2``); otherwise a role + agent-name
    based name.
    """
    if meta.label:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", meta.label).strip("_")
        return f"{slug}.md"
    base = meta.role.value.split("-")[-1]
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", meta.agent_name or "") if meta.agent_name else ""
    return f"{base}_{name}.md" if name else f"{base}.md"
