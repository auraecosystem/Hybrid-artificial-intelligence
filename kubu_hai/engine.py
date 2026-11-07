"""Backend-neutral hybrid inference orchestration.

Adapters are injected by applications; this module does not download models,
execute arbitrary code, or silently fall back to a different model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class InferenceRequest:
    task: str
    inputs: Mapping[str, Any]
    preferred_backend: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceResult:
    model_id: str
    backend: str
    output: Any
    latency_ms: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ModelAdapter(Protocol):
    model_id: str
    backend: str
    tasks: Sequence[str]

    def predict(self, request: InferenceRequest) -> Any: ...


class HybridEngine:
    """Selects a registered adapter deterministically and records provenance."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def infer(self, request: InferenceRequest) -> InferenceResult:
        import time

        if not request.task.strip():
            raise ValueError("task must be a non-empty string")
        adapter = self._registry.resolve(
            task=request.task, preferred_backend=request.preferred_backend
        )
        started = time.perf_counter()
        output = adapter.predict(request)
        elapsed = (time.perf_counter() - started) * 1000.0
        return InferenceResult(
            model_id=adapter.model_id,
            backend=adapter.backend,
            output=output,
            latency_ms=elapsed,
            metadata={"task": request.task, "requested_backend": request.preferred_backend},
        )
