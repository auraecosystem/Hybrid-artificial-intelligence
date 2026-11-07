"""PyTorch CPU, NVIDIA CUDA, and Apple Metal (MPS) device selection."""
from __future__ import annotations

from typing import Any


class DeviceUnavailableError(RuntimeError):
    """Raised when a requested accelerator is not usable."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for hardware detection; install kubu-hai-runtime[torch]"
        ) from exc
    return torch


def detect_devices(torch_module: Any | None = None) -> dict[str, bool]:
    """Return runtime-verified device availability, not just hardware presence."""
    torch = torch_module or _torch()
    cuda = bool(torch.cuda.is_available())
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps = bool(mps_backend and mps_backend.is_available() and mps_backend.is_built())
    return {"cpu": True, "cuda": cuda, "metal": mps}


def resolve_device(requested: str = "auto", torch_module: Any | None = None) -> tuple[str, Any]:
    """Resolve cpu/cuda/metal/auto to a PyTorch device and public backend name.

    Auto prefers CUDA, then Apple MPS, then CPU. Explicit accelerator requests
    fail closed; they never silently fall back to CPU.
    """
    torch = torch_module or _torch()
    choice = requested.lower().strip()
    if choice == "auto":
        available = detect_devices(torch)
        choice = "cuda" if available["cuda"] else "metal" if available["metal"] else "cpu"
    if choice == "cpu":
        return "cpu", torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise DeviceUnavailableError("CUDA requested but torch.cuda.is_available() is false")
        return "cuda", torch.device("cuda")
    if choice in {"metal", "mps", "apple-metal"}:
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if not (mps_backend and mps_backend.is_available() and mps_backend.is_built()):
            raise DeviceUnavailableError("Apple Metal requested but PyTorch MPS is unavailable")
        return "metal", torch.device("mps")
    raise ValueError("device must be one of: auto, cpu, cuda, metal")
