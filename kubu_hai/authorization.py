"""Public-key verified execution grants for the LMLM control-plane boundary.

KUBU-HAI verifies grants but never issues them: the LMLM control plane keeps
the Ed25519 private key and signs short-lived, task-scoped capabilities.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .engine import HybridEngine, InferenceRequest, InferenceResult


class AuthorizationError(PermissionError):
    """An execution request is not covered by a valid control-plane grant."""


@dataclass(frozen=True)
class ExecutionGrant:
    subject: str
    audience: str
    task: str
    expires_at: int
    not_before: int
    token_id: str
    backend: str | None = None


class GrantVerifier(Protocol):
    def verify(self, token: str) -> ExecutionGrant: ...


class ReplayStore(Protocol):
    def consume(self, token_id: str, expires_at: int, now: int) -> bool: ...


class InMemoryReplayStore:
    """Process-local replay protection; use a shared durable store in clusters."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}
        self._lock = threading.Lock()

    def consume(self, token_id: str, expires_at: int, now: int) -> bool:
        with self._lock:
            self._seen = {key: expiry for key, expiry in self._seen.items() if expiry > now}
            if token_id in self._seen:
                return False
            self._seen[token_id] = expires_at
            return True


def _decode_segment(segment: str) -> bytes:
    if not segment or any(ch.isspace() for ch in segment):
        raise AuthorizationError("malformed execution grant")
    try:
        return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
    except (ValueError, TypeError) as exc:
        raise AuthorizationError("malformed execution grant") from exc


class Ed25519GrantVerifier:
    """Verify compact payload.signature grants with an LMLM public key.

    The token payload is UTF-8 JSON. Its exact base64url payload segment is
    signed, avoiding re-serialization/canonicalization ambiguity.
    """

    def __init__(self, public_key: bytes) -> None:
        if len(public_key) != 32:
            raise ValueError("Ed25519 public key must be exactly 32 bytes")
        self._public_key = public_key

    def verify(self, token: str) -> ExecutionGrant:
        try:
            parts = token.split(".")\n            if len(parts) != 2:\n                raise AuthorizationError("malformed execution grant")\n            payload_segment, signature_segment = parts
            payload = _decode_segment(payload_segment)
            signature = _decode_segment(signature_segment)
            if len(signature) != 64:
                raise AuthorizationError("invalid execution grant signature")
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            Ed25519PublicKey.from_public_bytes(self._public_key).verify(
                signature, payload_segment.encode("ascii")
            )
            claims = json.loads(payload.decode("utf-8"))
            if not isinstance(claims, dict):
                raise AuthorizationError("execution grant claims must be an object")
            required = ("sub", "aud", "task", "exp", "nbf", "jti")
            if any(key not in claims for key in required):
                raise AuthorizationError("execution grant is missing required claims")
            if any(not isinstance(claims[key], str) or not claims[key].strip()
                   for key in ("sub", "aud", "task", "jti")):
                raise AuthorizationError("execution grant identity claims must be non-empty strings")
            if isinstance(claims["exp"], bool) or not isinstance(claims["exp"], int):
                raise AuthorizationError("execution grant exp must be an integer")
            if isinstance(claims["nbf"], bool) or not isinstance(claims["nbf"], int):
                raise AuthorizationError("execution grant nbf must be an integer")
            backend = claims.get("backend")
            if backend is not None and (not isinstance(backend, str) or not backend.strip()):
                raise AuthorizationError("execution grant backend must be a non-empty string")
            return ExecutionGrant(
                subject=claims["sub"], audience=claims["aud"], task=claims["task"],
                expires_at=claims["exp"], not_before=claims["nbf"],
                token_id=claims["jti"], backend=backend,
            )
        except AuthorizationError:
            raise
        except ImportError as exc:
            raise RuntimeError(
                "Ed25519 verification requires the security extra: pip install kubu-hai-runtime[security]"
            ) from exc
        except Exception as exc:
            raise AuthorizationError("execution grant signature or payload is invalid") from exc


class AuthorizedHybridExecutor:
    """Enforce LMLM-signed capabilities before invoking the inference engine."""

    def __init__(
        self,
        engine: HybridEngine,
        verifier: GrantVerifier,
        *,
        audience: str,
        replay_store: ReplayStore | None = None,
        clock: Any = time.time,
    ) -> None:
        if not audience.strip():
            raise ValueError("audience must be a non-empty string")
        self._engine = engine
        self._verifier = verifier
        self._audience = audience
        self._replay_store = replay_store or InMemoryReplayStore()
        self._clock = clock

    def infer(self, token: str, request: InferenceRequest) -> InferenceResult:
        grant = self._verifier.verify(token)
        now = int(self._clock())
        if grant.audience != self._audience:
            raise AuthorizationError("execution grant audience mismatch")
        if grant.not_before > now:
            raise AuthorizationError("execution grant is not active yet")
        if grant.expires_at <= now:
            raise AuthorizationError("execution grant has expired")
        if grant.task != request.task:
            raise AuthorizationError("execution grant does not authorize this task")
        if grant.backend is not None and grant.backend != request.preferred_backend:
            raise AuthorizationError("execution grant does not authorize this backend")
        if not self._replay_store.consume(grant.token_id, grant.expires_at, now):
            raise AuthorizationError("execution grant has already been used")
        result = self._engine.infer(request)
        metadata = dict(result.metadata)
        metadata.update({
            "authorized_subject": grant.subject,
            "authorization_audience": grant.audience,
            "authorization_token_id": grant.token_id,
        })
        return InferenceResult(
            model_id=result.model_id, backend=result.backend, output=result.output,
            latency_ms=result.latency_ms, metadata=metadata,
        )
