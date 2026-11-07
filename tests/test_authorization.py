"""Tests for the LMLM-signed execution grant boundary."""
import unittest

from kubu_hai.authorization import (
    AuthorizedHybridExecutor, AuthorizationError, ExecutionGrant,
    InMemoryReplayStore,
)
from kubu_hai.engine import HybridEngine, InferenceRequest
from kubu_hai.registry import ModelRegistry


class FakeVerifier:
    def __init__(self, grant):
        self.grant = grant

    def verify(self, token):
        if token != "valid-token":
            raise AuthorizationError("invalid")
        return self.grant


class EchoAdapter:
    model_id = "echo"
    backend = "cpu"
    tasks = ("text.echo",)

    def predict(self, request):
        return request.inputs["value"]


class AuthorizedExecutorTests(unittest.TestCase):
    def setUp(self):
        registry = ModelRegistry()
        registry.register(EchoAdapter())
        self.engine = HybridEngine(registry)
        self.now = 1_800_000_000
        self.grant = ExecutionGrant(
            subject="lmlm-agent-7", audience="kubu-hai-prod",
            task="text.echo", expires_at=self.now + 60,
            not_before=self.now - 1, token_id="grant-001", backend="cpu",
        )
        self.executor = AuthorizedHybridExecutor(
            self.engine, FakeVerifier(self.grant), audience="kubu-hai-prod",
            clock=lambda: self.now,
        )
        self.request = InferenceRequest(
            "text.echo", {"value": "hello"}, preferred_backend="cpu"
        )

    def test_authorized_request_executes_and_records_identity(self):
        result = self.executor.infer("valid-token", self.request)
        self.assertEqual(result.output, "hello")
        self.assertEqual(result.metadata["authorized_subject"], "lmlm-agent-7")
        self.assertEqual(result.metadata["authorization_token_id"], "grant-001")

    def test_replay_is_rejected(self):
        self.executor.infer("valid-token", self.request)
        with self.assertRaises(AuthorizationError):
            self.executor.infer("valid-token", self.request)

    def test_wrong_audience_is_rejected(self):
        executor = AuthorizedHybridExecutor(
            self.engine, FakeVerifier(self.grant), audience="other-runtime",
            clock=lambda: self.now,
        )
        with self.assertRaises(AuthorizationError):
            executor.infer("valid-token", self.request)

    def test_expired_grant_is_rejected(self):
        expired = ExecutionGrant(**{**self.grant.__dict__, "expires_at": self.now})
        executor = AuthorizedHybridExecutor(
            self.engine, FakeVerifier(expired), audience="kubu-hai-prod",
            clock=lambda: self.now,
        )
        with self.assertRaises(AuthorizationError):
            executor.infer("valid-token", self.request)

    def test_task_scope_is_enforced(self):
        request = InferenceRequest("admin.shell", {"value": "no"}, preferred_backend="cpu")
        with self.assertRaises(AuthorizationError):
            self.executor.infer("valid-token", request)

    def test_backend_scope_is_enforced(self):
        request = InferenceRequest("text.echo", {"value": "hello"}, preferred_backend="cuda")
        with self.assertRaises(AuthorizationError):
            self.executor.infer("valid-token", request)

    def test_future_grant_is_rejected(self):
        future = ExecutionGrant(**{**self.grant.__dict__, "not_before": self.now + 10})
        executor = AuthorizedHybridExecutor(
            self.engine, FakeVerifier(future), audience="kubu-hai-prod",
            clock=lambda: self.now,
        )
        with self.assertRaises(AuthorizationError):
            executor.infer("valid-token", self.request)

    def test_replay_store_expires_entries(self):
        store = InMemoryReplayStore()
        self.assertTrue(store.consume("one", 101, 100))
        self.assertFalse(store.consume("one", 101, 100))
        self.assertTrue(store.consume("one", 102, 101))


if __name__ == "__main__":
    unittest.main()
