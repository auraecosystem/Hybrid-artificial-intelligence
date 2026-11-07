"""Local, hash-verified model artifact manifests and PyTorch adapter loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .backends.pytorch import PyTorchAdapter, load_pytorch_adapter


class ArtifactIntegrityError(RuntimeError):
    pass


class ModelArtifactManager:
    """Loads artifacts from a trusted local root; does not fetch remote files."""
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("artifact root must be a directory")

    def _inside_root(self, relative: str) -> Path:
        requested = Path(relative)
        if requested.is_absolute() or ".." in requested.parts:
            raise ArtifactIntegrityError("absolute and parent-relative artifact paths are forbidden")
        try:
            candidate = (self.root / requested).resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ArtifactIntegrityError("artifact path does not resolve to an existing file") from exc
        if not candidate.is_relative_to(self.root) or not candidate.is_file():
            raise ArtifactIntegrityError("artifact path escapes root or is not a file")
        return candidate

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_pytorch(
        self,
        manifest_path: str,
        *,
        model_factory: Callable[[], Any],
        device: str = "auto",
    ) -> PyTorchAdapter:
        manifest_file = self._inside_root(manifest_path)
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError("manifest must be valid UTF-8 JSON") from exc
        required = {"model_id", "tasks", "weights", "sha256"}
        if not isinstance(manifest, dict) or not required.issubset(manifest):
            raise ArtifactIntegrityError(f"manifest requires fields: {sorted(required)}")
        if not isinstance(manifest["tasks"], list) or not manifest["tasks"]:
            raise ArtifactIntegrityError("manifest tasks must be a non-empty list")
        weights = self._inside_root(str(manifest["weights"]))
        actual = self._sha256(weights)
        expected = str(manifest["sha256"]).lower()
        if len(expected) != 64 or actual != expected:
            raise ArtifactIntegrityError("model weights SHA-256 does not match manifest")
        return load_pytorch_adapter(
            model_factory=model_factory,
            weights_path=weights,
            model_id=str(manifest["model_id"]),
            tasks=manifest["tasks"],
            device=device,
            input_mode=str(manifest.get("input_mode", "kwargs")),
        )
