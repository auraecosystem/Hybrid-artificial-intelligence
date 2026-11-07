import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "build.py"
SPEC = importlib.util.spec_from_file_location("kubu_hai_build", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BackendResolutionTests(unittest.TestCase):
    def setUp(self):
        self.caps = {
            "cmake": True, "ninja": True, "docker": True,
            "cuda": False, "metal": False, "linux": True, "macos": False,
        }

    def test_auto_defaults_to_cpu(self):
        self.assertEqual(MODULE.resolve_backend("auto", self.caps), "cpu")

    def test_cpu_is_always_available(self):
        self.assertEqual(MODULE.resolve_backend("cpu", self.caps), "cpu")

    def test_cuda_requires_driver_probe(self):
        with self.assertRaisesRegex(RuntimeError, "nvidia-smi"):
            MODULE.resolve_backend("cuda", self.caps)

    def test_metal_requires_macos(self):
        with self.assertRaisesRegex(RuntimeError, "non-macOS"):
            MODULE.resolve_backend("metal", self.caps)

    def test_cuda_is_allowed_when_detected(self):
        self.caps["cuda"] = True
        self.assertEqual(MODULE.resolve_backend("cuda", self.caps), "cuda")

    def test_metal_is_allowed_on_macos(self):
        self.caps["metal"] = True
        self.assertEqual(MODULE.resolve_backend("metal", self.caps), "metal")


if __name__ == "__main__":
    unittest.main()
