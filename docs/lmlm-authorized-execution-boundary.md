# LMLM Authorized Execution Boundary

KUBU-HAI is an execution subsystem. LMLM retains identity, policy, authorization, and orchestration authority. This document defines how a task-scoped authorization grant is checked before KUBU-HAI invokes a registered model adapter.

## Trust model

- LMLM holds an Ed25519 private signing key and issues short-lived execution grants.
- KUBU-HAI receives only the corresponding public key and verifies grants.
- KUBU-HAI does not issue grants, make policy decisions, or load a private signing key.
- The grant authorizes a narrow task, audience, validity window, and optionally one backend.
- The grant is not transport encryption. Network deployments must use authenticated transport such as mutual TLS.

## Install

```bash
python -m pip install -e '.[security]'
```

## Token format

The compact token format is `base64url(UTF-8 JSON claims).base64url(Ed25519 signature)`. The signature covers the exact encoded claims segment. Required claims:

| Claim | Meaning |
| --- | --- |
| `sub` | Authorized LMLM agent or principal |
| `aud` | Exact KUBU-HAI runtime audience |
| `task` | Exact task name allowed by the grant |
| `nbf`, `exp` | Integer activation and expiration times |
| `jti` | Unique, single-use grant identifier |
| `backend` | Optional exact backend constraint: `cpu`, `cuda`, or `metal` |

## Runtime integration

```python
from kubu_hai import (
    AuthorizedHybridExecutor,
    Ed25519GrantVerifier,
    HybridEngine,
    InferenceRequest,
    InMemoryReplayStore,
)

executor = AuthorizedHybridExecutor(
    HybridEngine(registry),
    Ed25519GrantVerifier(lmlm_ed25519_public_key_bytes),
    audience="kubu-hai-prod",
    replay_store=InMemoryReplayStore(),
)

result = executor.infer(
    lmlm_signed_capability,
    InferenceRequest("text.generate", {"prompt": "Hello"}, preferred_backend="cpu"),
)
```

The token is created and signed by LMLM, not by KUBU-HAI. The executor checks signature (through the verifier), audience, task, optional backend, activation/expiry, and replay status before inference.

## Replay protection and deployment

The included replay store is process-local and intended for development or a single runtime process. A multi-worker or multi-host deployment MUST inject a shared, atomic, durable replay store so a grant cannot be replayed on another worker. A Redis-backed implementation is one possible deployment adapter; it is not included in this change.

Never expose a public unauthenticated inference endpoint. Add authenticated transport, rate limits, request-size limits, and operational audit logging at the service boundary. Keep the LMLM signing key in the control plane's secret-management boundary.

## Tests

The authorization unit tests cover successful execution, replay rejection, audience mismatch, task/backend scope, and activation/expiry. The optional cryptographic verifier requires the `security` extra.
