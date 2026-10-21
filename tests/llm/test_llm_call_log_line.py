"""Every provider attempt leaves one INFO line with model and token counts.

The line is the operator's cost receipt: it names the model and effort that
actually ran (primary or fallback) and the provider-reported token usage.
"""

from __future__ import annotations

import logging
import time

import pytest

from src.config import ModelConfig
from src.llm.backend import CompletionResult as BackendCompletionResult
from src.llm.executor import _emit_call_telemetry  # pyright: ignore[reportPrivateUsage]
from src.llm.runtime import AttemptPlan
from src.llm.types import LLMTelemetryContext


def _plan(*, is_fallback: bool = False, attempt: int = 1) -> AttemptPlan:
    # Production callers pass no per-call effort, so the plan carries None and
    # the configured effort lives on selected_config. The log must report it.
    return AttemptPlan(
        provider="openai",
        model="meta/muse-spark-1.2-contributor",
        client=object(),  # pyright: ignore[reportArgumentType]
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=ModelConfig(
            model="meta/muse-spark-1.2-contributor",
            transport="openai",
            thinking_effort="medium",
        ),
        attempt=attempt,
        retry_attempts=3,
        is_fallback=is_fallback,
    )


def test_success_logs_model_effort_and_tokens(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="src.llm.executor"):
        _emit_call_telemetry(
            plan=_plan(),
            telemetry=LLMTelemetryContext(call_purpose="deriver.representation"),
            provider="openai",
            model="meta/muse-spark-1.2-contributor",
            max_tokens=16000,
            started_at=time.perf_counter(),
            messages=[],
            tools=None,
            tool_choice=None,
            was_stream=False,
            result=BackendCompletionResult(
                content="{}",
                input_tokens=1200,
                output_tokens=340,
                cache_read_input_tokens=800,
            ),
            error=None,
        )

    lines = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("LLM_CALL")
    ]
    assert len(lines) == 1
    line = lines[0]
    assert "purpose=deriver.representation" in line
    assert "model=meta/muse-spark-1.2-contributor" in line
    assert "effort=medium" in line
    assert "attempt=1/3" in line
    assert "fallback=false" in line
    assert "outcome=success" in line
    assert "input_tokens=1200" in line
    assert "cache_read_tokens=800" in line
    assert "output_tokens=340" in line


def test_failed_fallback_attempt_logs_error_class(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="src.llm.executor"):
        _emit_call_telemetry(
            plan=_plan(is_fallback=True, attempt=3),
            telemetry=None,
            provider="openai",
            model="grok-4.3",
            max_tokens=16000,
            started_at=time.perf_counter(),
            messages=[],
            tools=None,
            tool_choice=None,
            was_stream=False,
            result=None,
            error=TimeoutError("upstream timed out"),
        )

    lines = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("LLM_CALL")
    ]
    assert len(lines) == 1
    line = lines[0]
    assert "model=grok-4.3" in line
    assert "fallback=true" in line
    assert "outcome=error" in line
    assert "error=TimeoutError" in line
    assert "input_tokens=0" in line
