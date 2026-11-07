# KUBU-HAI Hardware Backends and Model Artifacts

## Supported execution targets

| Public backend | PyTorch device | Requirement |
|---|---|---|
| `cpu` | `cpu` | PyTorch CPU build |
| `cuda` | `cuda` | NVIDIA GPU, compatible driver, CUDA-enabled PyTorch |
| `metal` | `mps` | Apple Silicon/macOS and PyTorch MPS build |
| `auto` | selected at runtime | CUDA, then MPS, then CPU |

The adapter runs actual `torch.nn.Module` inference, moves tensor inputs to the
selected device, uses `torch.inference_mode()`, and returns detached host values.
Explicit CUDA/Metal requests raise `DeviceUnavailableError` if unavailable;
there is no silent accelerator-to-CPU fallback. `auto` is the only mode that
selects a fallback by design.

## Model artifact manifest

Keep a JSON manifest and weights file under a local artifact root. Example:

```json
{
  "model_id": "example-mlp-v1",
  "tasks": ["tabular.regression"],
  "weights": "example-mlp-v1/state_dict.pt",
  "sha256": "REPLACE_WITH_64_CHARACTER_SHA256",
  "input_mode": "kwargs"
}
```

Compute the digest with `sha256sum state_dict.pt`. The artifact manager checks
that manifest and weight paths remain inside the configured root and verifies
the SHA-256 before loading. PyTorch loading uses `weights_only=True`; the
architecture is supplied by trusted application code through `model_factory`.
No arbitrary architecture imports, pickle fallback, or remote model downloads
are performed. Only state-dict checkpoints are supported by this loader.

## Usage

```python
from kubu_hai.artifacts import ModelArtifactManager
from kubu_hai.engine import HybridEngine, InferenceRequest
from kubu_hai.registry import ModelRegistry

# Define this architecture in trusted, reviewed application code.
def build_model():
    import torch
    return torch.nn.Linear(4, 2)

artifacts = ModelArtifactManager("./model-artifacts")
adapter = artifacts.load_pytorch(
    "linear/model.json", model_factory=build_model, device="auto"
)
registry = ModelRegistry()
registry.register(adapter)
engine = HybridEngine(registry)
result = engine.infer(InferenceRequest(
    task="tabular.classification", inputs={"input": [[0.1, 0.2, 0.3, 0.4]]},
    preferred_backend=adapter.backend,
))
print(result.model_id, result.backend, result.output)
```

For a model whose forward signature is `forward(input)`, set manifest
`input_mode` to `single`; the request must contain exactly one named input.

## Validation and hardware CI

- The normal CI job installs CPU PyTorch and runs a real tensor inference test.
- Optional self-hosted CUDA and Apple Silicon runners can execute the same
  hardware tests when labels/secrets are configured by repository maintainers.
- Hardware availability tests skip unavailable physical devices; a skipped
  accelerator test is not evidence that the accelerator passed.

Run locally:

```bash
python -m pip install -e '.[torch]'
python -m compileall -q kubu_hai tests
python -m unittest discover -s tests -v
```

## Limits before production

This adapter is a single-process PyTorch inference integration. Production
serving still needs per-task input/output schemas, batching, concurrency and
cancellation policy, resource quotas, model signing/attestation, metrics,
load/quality tests, and hardware-specific performance benchmarks. CUDA and MPS
execution must be validated on actual target hardware. The local SHA-256 digest
provides integrity relative to the manifest, not publisher authenticity; sign
manifests in a later release workflow.
