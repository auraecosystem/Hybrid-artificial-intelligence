"""KUBU-HAI hybrid inference foundation."""
from .engine import HybridEngine, InferenceRequest, InferenceResult
from .registry import ModelRegistry
from .artifacts import ArtifactIntegrityError, ModelArtifactManager
from .backends.pytorch import PyTorchAdapter, load_pytorch_adapter
from .authorization import (
    AuthorizedHybridExecutor, AuthorizationError, Ed25519GrantVerifier,
    ExecutionGrant, InMemoryReplayStore,
)

__all__ = [
    "HybridEngine", "InferenceRequest", "InferenceResult", "ModelRegistry",
    "ArtifactIntegrityError", "ModelArtifactManager",
    "PyTorchAdapter", "load_pytorch_adapter",
    "AuthorizedHybridExecutor", "AuthorizationError", "Ed25519GrantVerifier",
    "ExecutionGrant", "InMemoryReplayStore",
]
