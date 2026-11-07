"""In-memory model adapter registry."""
from __future__ import annotations

from typing import Any


class ModelNotFoundError(LookupError):
    pass


class ModelRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Any] = {}

    def register(self, adapter: Any) -> None:
        model_id = getattr(adapter, "model_id", None)
        backend = getattr(adapter, "backend", None)
        tasks = getattr(adapter, "tasks", None)
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("adapter.model_id must be a non-empty string")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("adapter.backend must be a non-empty string")
        if not tasks or not all(isinstance(task, str) and task for task in tasks):
            raise ValueError("adapter.tasks must contain task names")
        if not callable(getattr(adapter, "predict", None)):
            raise ValueError("adapter.predict must be callable")
        if model_id in self._adapters:
            raise ValueError(f"model already registered: {model_id}")
        self._adapters[model_id] = adapter

    def resolve(self, task: str, preferred_backend: str | None = None) -> Any:
        candidates = [
            adapter for adapter in self._adapters.values()
            if task in adapter.tasks
        ]
        if preferred_backend is not None:
            candidates = [a for a in candidates if a.backend == preferred_backend]
        if not candidates:
            raise ModelNotFoundError(
                f"no model registered for task={task!r}, backend={preferred_backend!r}"
            )
        # Stable selection: registration order is preserved by dict.
        return candidates[0]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"model_id": a.model_id, "backend": a.backend, "tasks": list(a.tasks)}
            for a in self._adapters.values()
        ]
