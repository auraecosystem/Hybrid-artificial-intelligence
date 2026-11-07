# KUBU-HAI Hybrid Runtime Foundation

## Scope

This directory adds an isolated, backend-neutral inference foundation without
rewriting the repository's existing experiments, Flutter runner, or legacy
workflows. It is not a trained model and does not claim machine awareness.

## Runtime flow

1. The application constructs an `InferenceRequest` with a task and inputs.
2. The application registers trusted model adapters in `ModelRegistry`.
3. `HybridEngine` resolves an adapter by task and optional exact backend.
4. The adapter performs inference; the engine returns output, latency, and
   model/backend provenance.

No implicit network downloads, arbitrary plugin execution, or silent backend
fallbacks occur. A missing task/backend raises `ModelNotFoundError`.

## Adapter contract

An adapter exposes:

- `model_id: str`
- `backend: str` (for example `cpu`, `cuda`, `rocm`, or `metal`)
- `tasks: Sequence[str]`
- `predict(request: InferenceRequest) -> Any`

Framework-specific integrations (PyTorch, TensorFlow, ONNX Runtime, classical
ML, audio, and vision) should live in separate adapter modules. Keep framework
imports lazy and optional so the orchestration core remains lightweight.

## Local validation

```bash
python -m compileall -q kubu_hai tests
python -m unittest discover -s tests -v
```

## Next production gates

- Add schema validation and explicit input/output contracts per task.
- Add model artifact manifests, hashes, signatures, provenance, and safe loading.
- Add cancellation, deadlines, resource limits, structured logging, and metrics.
- Implement and test CPU/CUDA/ROCm/Metal adapters on their actual hardware.
- Add evaluation datasets, quality thresholds, load tests, and security review.
- Integrate the artifact download workflow only after a signed model-artifact
  manifest and retention policy are defined.

LMLM remains a separate model/runtime. Interoperability should be through a
versioned adapter contract, not by merging project identities or codebases.
