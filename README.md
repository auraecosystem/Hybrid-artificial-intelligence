# KUBU-HAI Hybrid Runtime

KUBU-HAI is a backend-neutral inference runtime foundation for LMLM. It provides a model-adapter registry, deterministic task routing, execution provenance, local model-artifact verification, and a PyTorch adapter for CPU, NVIDIA CUDA, and Apple Metal (MPS).

## Runtime boundary

LMLM remains the sovereign control plane. It owns identity, authorization, policy, task state, memory, and orchestration. KUBU-HAI is an execution subsystem: it receives an already-authorized inference request, selects a registered adapter, executes the model, and returns output plus backend/model/latency provenance.

The runtime does not acquire authority from a remote model or adapter. It does not download model weights, execute code from a manifest, or silently switch backends after an explicit accelerator request fails.

## Install

Python 3.10 or newer:

```bash
python -m pip install -e .
```

For real PyTorch inference:

```bash
python -m pip install -e '.[torch]'
```

## Backend behavior

| Request | Behavior |
| --- | --- |
| `cpu` | Always-available PyTorch CPU execution |
| `cuda` | Uses CUDA only when `torch.cuda.is_available()` is true; otherwise fails closed |
| `metal` | Uses Apple MPS only when it is built and available; otherwise fails closed |
| `auto` | Selects CUDA, then Apple MPS, then CPU |

Backend selection is performed at runtime by PyTorch. A build host merely having `nvidia-smi` or running macOS is not proof that a usable accelerator is available.

## Minimal usage

```python
import torch

from kubu_hai import HybridEngine, InferenceRequest, ModelRegistry, PyTorchAdapter

model = torch.nn.Linear(2, 1, bias=False)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[2.0, 3.0]]))

adapter = PyTorchAdapter(
    model,
    model_id="linear-demo",
    tasks=["tensor.regression"],
    device="auto",
    input_mode="single",
)
registry = ModelRegistry()
registry.register(adapter)

engine = HybridEngine(registry)
result = engine.infer(
    InferenceRequest("tensor.regression", {"features": [[1.0, 2.0]]})
)

print(result.output)   # 8.0
print(result.backend)  # cpu, cuda, or metal
print(result.latency_ms)
```

For LMLM integration, the control plane should authorize the task and construct the `InferenceRequest`; it should not delegate policy decisions to the backend adapter. The returned `InferenceResult` carries the model ID, selected backend, latency, and request metadata for LMLM's audit/event pipeline.

## Verified local model artifacts

`ModelArtifactManager` accepts a local manifest and a local weights file beneath a configured trusted root. It validates the SHA-256 digest and rejects absolute paths, parent traversal, symlink escapes, and invalid manifests before loading weights. Model architecture must be supplied by trusted application code; checkpoints are loaded with `weights_only=True`.

Example manifest:

```json
{
  "model_id": "my-model",
  "tasks": ["tensor.regression"],
  "weights": "weights/model.pt",
  "sha256": "REPLACE_WITH_THE_64_CHARACTER_SHA256_DIGEST",
  "input_mode": "single"
}
```

## Tests

```bash
python -m pip install -e .
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -p 'test_[a-z]*.py' -v
python -m unittest discover -s kubu-hai-build/tests -v
```

The CPU CI job installs CPU-only PyTorch so that the real CPU inference path is exercised rather than skipped. CUDA and Apple MPS execution must be validated on physical/virtual runners with working accelerator drivers; the manually dispatched hardware workflow uses self-hosted runner labels `cuda` and `metal`.

## Build tooling

The root CMake configuration is a Flutter Linux desktop runner. The separate `kubu-hai-build/build.py` tool validates that project's generated inputs and supports native Flutter/Linux or generic CMake builds. It does not compile CUDA kernels or Metal shaders. CUDA/MPS inference in this repository is provided by the PyTorch runtime adapter, not by the Flutter CMake runner.

See [kubu-hai-build/README.md](kubu-hai-build/README.md) for build-driver details.
