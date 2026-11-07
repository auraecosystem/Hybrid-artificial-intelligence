import unittest

from kubu_hai.engine import HybridEngine, InferenceRequest
from kubu_hai.registry import ModelNotFoundError, ModelRegistry


class EchoAdapter:
    model_id = "echo-v1"
    backend = "cpu"
    tasks = ("text.echo",)

    def predict(self, request):
        return {"echo": request.inputs["text"]}


class HybridEngineTests(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry()
        self.registry.register(EchoAdapter())
        self.engine = HybridEngine(self.registry)

    def test_inference_records_adapter_provenance(self):
        result = self.engine.infer(InferenceRequest("text.echo", {"text": "hello"}))
        self.assertEqual(result.output, {"echo": "hello"})
        self.assertEqual(result.model_id, "echo-v1")
        self.assertEqual(result.backend, "cpu")
        self.assertGreaterEqual(result.latency_ms, 0)

    def test_backend_constraint_is_enforced(self):
        with self.assertRaises(ModelNotFoundError):
            self.engine.infer(InferenceRequest("text.echo", {"text": "x"}, "cuda"))

    def test_unknown_task_fails_closed(self):
        with self.assertRaises(ModelNotFoundError):
            self.engine.infer(InferenceRequest("vision.classify", {}))

    def test_empty_task_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.infer(InferenceRequest(" ", {}))

    def test_duplicate_model_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.registry.register(EchoAdapter())


if __name__ == "__main__":
    unittest.main()
