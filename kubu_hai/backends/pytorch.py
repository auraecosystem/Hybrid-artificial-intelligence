"""Real PyTorch model adapter for CPU, CUDA, and Apple Metal (MPS)."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Sequence

from .device import resolve_device


def _move_inputs(value: Any, torch: Any, device: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _move_inputs(item, torch, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_inputs(item, torch, device) for item in value)
    if isinstance(value, list):
        # Strings and other non-numeric lists are not tensors.
        try:
            return torch.as_tensor(value, device=device)
        except (TypeError, ValueError, RuntimeError):
            return [_move_inputs(item, torch, device) for item in value]
    if isinstance(value, (bool, int, float)):
        return torch.as_tensor(value, device=device)
    if hasattr(value, "to") and callable(value.to):
        return value.to(device)
    return value


def _to_host(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_host(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_host(item) for item in value)
    if isinstance(value, list):
        return [_to_host(item) for item in value]
    if hasattr(value, "detach") and callable(value.detach):
        tensor = value.detach().to("cpu")
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    return value


class PyTorchAdapter:
    """Adapter for an already-constructed torch.nn.Module.

    input_mode='kwargs' calls model(**request.inputs); input_mode='single'
    requires exactly one input and calls model(input). No model code is loaded
    dynamically and no network access is performed.
    """
    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        tasks: Sequence[str],
        device: str = "auto",
        input_mode: str = "kwargs",
        torch_module: Any | None = None,
    ) -> None:
        try:
            import torch as imported_torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required; install kubu-hai-runtime[torch]") from exc
        self._torch = torch_module or imported_torch
        if not isinstance(model, self._torch.nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        if not model_id.strip() or not tasks:
            raise ValueError("model_id and tasks must be non-empty")
        if input_mode not in {"kwargs", "single"}:
            raise ValueError("input_mode must be 'kwargs' or 'single'")
        self.model_id = model_id
        self.tasks = tuple(tasks)
        self.backend, self.device = resolve_device(device, self._torch)
        self.input_mode = input_mode
        self.model = model.to(self.device)
        self.model.eval()

    def predict(self, request: Any) -> Any:
        inputs = _move_inputs(request.inputs, self._torch, self.device)
        with self._torch.inference_mode():
            if self.input_mode == "single":
                if not isinstance(inputs, Mapping) or len(inputs) != 1:
                    raise ValueError("single input mode requires exactly one named input")
                output = self.model(next(iter(inputs.values())))
            else:
                if not isinstance(inputs, Mapping):
                    raise TypeError("kwargs input mode requires request.inputs to be a mapping")
                output = self.model(**inputs)
        return _to_host(output)


def load_pytorch_adapter(
    *,
    model_factory: Callable[[], Any],
    weights_path: str | Path,
    model_id: str,
    tasks: Sequence[str],
    device: str = "auto",
    input_mode: str = "kwargs",
) -> PyTorchAdapter:
    """Construct architecture via trusted code and load a weights-only checkpoint.

    Checkpoints must contain a plain state_dict. Arbitrary pickle deserialization
    is disabled with weights_only=True; model architecture is never imported from
    a manifest or checkpoint.
    """
    import torch

    path = Path(weights_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("weights_path must identify a regular file")
    model = model_factory()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint must contain a non-empty state_dict mapping")
    model.load_state_dict(state, strict=True)
    return PyTorchAdapter(
        model, model_id=model_id, tasks=tasks, device=device,
        input_mode=input_mode, torch_module=torch,
    )
