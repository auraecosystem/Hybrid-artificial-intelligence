import unittest

try:
    import torch
except ImportError:
    torch = None

from kubu_hai.backends.device import DeviceUnavailableError, detect_devices, resolve_device
from kubu_hai.backends.pytorch import PyTorchAdapter
from kubu_hai.engine import HybridEngine, InferenceRequest
from kubu_hai.registry import ModelRegistry


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class PyTorchBackendTests(unittest.TestCase):
    class Linear(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                self.proj.weight.copy_(torch.tensor([[2.0, 3.0]]))

        def forward(self, features):
            return self.proj(features)

    def _run_model(self, backend):
        adapter = PyTorchAdapter(
            self.Linear(), model_id=f"linear-{backend}",
            tasks=["tensor.regression"], device=backend,
        )
        registry = ModelRegistry()
        registry.register(adapter)
        result = HybridEngine(registry).infer(InferenceRequest(
            "tensor.regression", {"features": [[1.0, 2.0]]}, backend
        ))
        self.assertEqual(result.backend, backend)
        # Scalar tensors are deliberately normalized to Python scalars.
        self.assertAlmostEqual(result.output, 8.0, places=4)

    def test_cpu_executes_real_torch_model_through_hybrid_engine(self):
        self._run_model("cpu")

    def test_available_accelerators_execute_real_inference(self):
        devices = detect_devices()
        exercised = False
        for backend in ("cuda", "metal"):
            if devices[backend]:
                self._run_model(backend)
                exercised = True
        if not exercised:
            self.skipTest("No CUDA or Apple MPS accelerator on this runner")

    def test_device_detection_reports_cpu(self):
        self.assertTrue(detect_devices()["cpu"])

    def test_explicit_unavailable_cuda_fails_closed(self):
        if torch.cuda.is_available():
            self.skipTest("CUDA is available on this runner")
        with self.assertRaises(DeviceUnavailableError):
            resolve_device("cuda")

    def test_explicit_unavailable_metal_fails_closed(self):
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available() and mps.is_built():
            self.skipTest("MPS is available on this runner")
        with self.assertRaises(DeviceUnavailableError):
            resolve_device("metal")

    def test_auto_selects_an_available_backend(self):
        backend, device = resolve_device("auto")
        self.assertIn(backend, {"cpu", "cuda", "metal"})
        self.assertEqual(str(device), {"cpu": "cpu", "cuda": "cuda", "metal": "mps"}[backend])


if __name__ == "__main__":
    unittest.main()
