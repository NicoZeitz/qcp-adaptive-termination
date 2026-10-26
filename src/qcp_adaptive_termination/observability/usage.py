"""Per-query usage and cost tracking for LLM and embedding calls.

Records every call and computes USD cost from configured per-model pricing."""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import uuid4

from qcp_adaptive_termination.models import AgentRole, Config, ModelCost

_PER_MILLION = Decimal(1_000_000)


@dataclass(slots=True)
class EmbeddingCall:
    """A single distinct embedding computation (one unique text)."""

    call_id: str
    text: str
    model: str
    prompt_tokens: int


@dataclass(slots=True)
class LLMCall:
    """A single LLM run (role/agent label + aggregated SDK usage)."""

    call_id: str
    role: AgentRole | str
    agent_name: str | None
    label: str | None
    model: str
    requests: int
    input_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int


def _embedding_cost(model_cost: ModelCost | None, tokens: int) -> Decimal:
    """Compute the USD cost of an embedding computation.

    Args:
        model_cost: pricing for the model, or None to return zero
        tokens: the number of prompt tokens

    Returns:
        The cost in dollars
    """
    if model_cost is None or model_cost.embedding is None:
        return Decimal(0)
    return Decimal(tokens) * model_cost.embedding / _PER_MILLION


def _llm_cost(model_cost: ModelCost | None, call: LLMCall) -> Decimal:
    """Compute the USD cost of an LLM call from its token usage.

    Args:
        model_cost: pricing for the model, or None to return zero
        call: the LLM call with raw token counts

    Returns:
        The cost in dollars
    """
    if model_cost is None:
        return Decimal(0)
    non_cached_input = call.input_tokens - call.cached_input_tokens - call.cache_write_tokens
    total = (
        Decimal(non_cached_input) * (model_cost.input or Decimal(0))
        + Decimal(call.cached_input_tokens) * (model_cost.cached_input or Decimal(0))
        + Decimal(call.cache_write_tokens) * (model_cost.cache_write or Decimal(0))
        + Decimal(call.output_tokens) * (model_cost.output or Decimal(0))
    )
    return total / _PER_MILLION


CSV_COLUMNS = [
    "type",
    "call_id",
    "model",
    "label",
    "role",
    "agent_name",
    "text",
    "requests",
    "prompt_tokens",
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cost",
]


def embedding_csv_row(config: Config, model: str, prompt_tokens: int, call_id: str, text: str = "") -> dict[str, Any]:
    """Build a CSV row for a distinct embedding computation.

    Args:
        config: pricing for cost computation
        model: the embedding model
        prompt_tokens: prompt token count
        call_id: unique call identifier
        text: the embedded text, for reference

    Returns:
        A row dict matching ``CSV_COLUMNS``
    """
    cost = _embedding_cost(config.costs.get(model), prompt_tokens)
    return {
        "type": "embedding",
        "call_id": call_id,
        "model": model,
        "label": "",
        "role": "",
        "agent_name": "",
        "text": text,
        "requests": 1,
        "prompt_tokens": prompt_tokens,
        "input_tokens": prompt_tokens,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cost": float(cost),
    }


def llm_csv_row(
    config: Config,
    role: AgentRole | str,
    agent_name: str | None,
    label: str | None,
    model: str,
    call_id: str,
    *,
    requests: int,
    input_tokens: int,
    cached_input_tokens: int,
    cache_write_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
) -> dict[str, Any]:
    """Build a CSV row for a single LLM run.

    Args:
        config: pricing for cost computation
        role: the agent role
        agent_name: optional agent display name
        label: optional run label
        model: the model used
        call_id: unique call identifier
        requests: number of API requests in the run
        input_tokens: input token count
        cached_input_tokens: input tokens served from cache
        cache_write_tokens: input tokens written to cache
        output_tokens: output token count
        reasoning_tokens: reasoning token count

    Returns:
        A row dict matching ``CSV_COLUMNS``
    """
    call = LLMCall(
        call_id=call_id,
        role=role,
        agent_name=agent_name,
        label=label,
        model=model,
        requests=requests,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    return {
        "type": "llm",
        "call_id": call_id,
        "model": model,
        "label": label or "",
        "role": role.value if isinstance(role, AgentRole) else str(role),
        "agent_name": agent_name or "",
        "text": "",
        "requests": requests,
        "prompt_tokens": input_tokens,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_tokens": cache_write_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost": float(_llm_cost(config.costs.get(model), call)),
    }


@dataclass(slots=True)
class ModelTotals:
    """Per-model accumulated usage (cost as Decimal until serialized)."""

    calls: int = 0
    requests: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost: Decimal = Decimal(0)


@dataclass(slots=True)
class UsageReport:
    """Financial-ish summary of one query execution."""

    embeddings: list[EmbeddingCall] = field(default_factory=list)
    llm: list[LLMCall] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)

    def record_embedding(self, text: str, model: str, prompt_tokens: int) -> bool:
        """Record one embedding computation, deduplicated by text.

        Args:
            text: the embedded text
            model: the embedding model
            prompt_tokens: prompt token count for the text

        Returns:
            False if text was already recorded
        """
        if any(call.text == text for call in self.embeddings):
            return False
        self.embeddings.append(EmbeddingCall(call_id=uuid4().hex, text=text, model=model, prompt_tokens=prompt_tokens))
        return True

    def record_llm(
        self,
        role: AgentRole | str,
        agent_name: str | None,
        label: str | None,
        model: str,
        usage,
    ) -> None:
        """Record one LLM run from an SDK usage object plus the model it ran on.

        Args:
            role: the agent role that made the run
            agent_name: optional agent display name
            label: optional run label
            model: the model used
            usage: the SDK Usage object with token counts
        """
        self.llm.append(
            LLMCall(
                call_id=uuid4().hex,
                role=role,
                agent_name=agent_name,
                label=label,
                model=model,
                requests=usage.requests,
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.input_tokens_details.cached_tokens,
                cache_write_tokens=getattr(usage.input_tokens_details, "cache_write_tokens", 0),
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
            )
        )

    def _embedding_totals(self, costs: dict[str, ModelCost]) -> dict[str, ModelTotals]:
        """Aggregate embedding usage per model.

        Args:
            costs: pricing keyed by model name

        Returns:
            Model name to accumulated totals
        """
        totals: dict[str, ModelTotals] = defaultdict(ModelTotals)
        for call in self.embeddings:
            t = totals[call.model]
            t.calls += 1
            t.input_tokens += call.prompt_tokens
            t.cost += _embedding_cost(costs.get(call.model), call.prompt_tokens)
        return dict(totals)

    def _llm_totals(self, costs: dict[str, ModelCost]) -> dict[str, ModelTotals]:
        """Aggregate LLM usage per model.

        Args:
            costs: pricing keyed by model name

        Returns:
            Model name to accumulated totals
        """
        totals: dict[str, ModelTotals] = defaultdict(ModelTotals)
        for call in self.llm:
            t = totals[call.model]
            t.calls += 1
            t.requests += call.requests
            t.input_tokens += call.input_tokens
            t.cached_input_tokens += call.cached_input_tokens
            t.cache_write_tokens += call.cache_write_tokens
            t.output_tokens += call.output_tokens
            t.reasoning_tokens += call.reasoning_tokens
            t.cost += _llm_cost(costs.get(call.model), call)
        return dict(totals)

    def call_rows(self, config: Config) -> list[dict[str, Any]]:
        """Build one row per call (embeddings then LLM) for the per-call CSV.

        Args:
            config: pricing for cost computation

        Returns:
            A list of row dicts matching ``CSV_COLUMNS``
        """
        rows = [embedding_csv_row(config, c.model, c.prompt_tokens, c.call_id, c.text) for c in self.embeddings]
        rows.extend(
            llm_csv_row(
                config,
                c.role,
                c.agent_name,
                c.label,
                c.model,
                c.call_id,
                requests=c.requests,
                input_tokens=c.input_tokens,
                cached_input_tokens=c.cached_input_tokens,
                cache_write_tokens=c.cache_write_tokens,
                output_tokens=c.output_tokens,
                reasoning_tokens=c.reasoning_tokens,
            )
            for c in self.llm
        )
        return rows

    def summarize(self, config: Config) -> dict:
        """Build a JSON-serializable usage report (money as float).

        Includes aggregated totals per model and the full per-call breakdown.

        Args:
            config: pricing for cost computation

        Returns:
            A JSON-ready dict with time, embeddings, llm and total cost
        """
        embedding_totals = self._embedding_totals(config.costs)
        llm_totals = self._llm_totals(config.costs)
        embedding_calls = [
            embedding_csv_row(config, c.model, c.prompt_tokens, c.call_id, c.text) for c in self.embeddings
        ]
        llm_calls = [
            llm_csv_row(
                config,
                c.role,
                c.agent_name,
                c.label,
                c.model,
                c.call_id,
                requests=c.requests,
                input_tokens=c.input_tokens,
                cached_input_tokens=c.cached_input_tokens,
                cache_write_tokens=c.cache_write_tokens,
                output_tokens=c.output_tokens,
                reasoning_tokens=c.reasoning_tokens,
            )
            for c in self.llm
        ]

        def _embedding_row(t: ModelTotals) -> dict:
            return {"calls": t.calls, "input_tokens": t.input_tokens, "cost": float(t.cost)}

        def _llm_row(t: ModelTotals) -> dict:
            return {
                "runs": t.calls,
                "requests": t.requests,
                "input_tokens": t.input_tokens,
                "cached_input_tokens": t.cached_input_tokens,
                "cache_write_tokens": t.cache_write_tokens,
                "output_tokens": t.output_tokens,
                "reasoning_tokens": t.reasoning_tokens,
                "cost": float(t.cost),
            }

        embedding_rows = {model: _embedding_row(t) for model, t in embedding_totals.items()}
        llm_rows = {model: _llm_row(t) for model, t in llm_totals.items()}
        embedding_cost = sum(t.cost for t in embedding_totals.values())
        llm_cost = sum(t.cost for t in llm_totals.values())

        return {
            "time_s": time.monotonic() - self.started_at,
            "embeddings": {
                "calls": sum(t.calls for t in embedding_totals.values()),
                "per_model": embedding_rows,
                "per_call": embedding_calls,
                "cost": float(embedding_cost),
            },
            "llm": {
                "runs": sum(t.calls for t in llm_totals.values()),
                "requests": sum(t.requests for t in llm_totals.values()),
                "per_model": llm_rows,
                "per_call": llm_calls,
                "cost": float(llm_cost),
            },
            "total_cost": float(embedding_cost + llm_cost),
        }
