import json
import tempfile
import unittest
from pathlib import Path

from kubu_hai.artifacts import ArtifactIntegrityError, ModelArtifactManager


class ArtifactManagerTests(unittest.TestCase):
    def test_rejects_hash_mismatch_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.bin").write_bytes(b"not-a-real-checkpoint")
            (root / "model.json").write_text(json.dumps({
                "model_id": "demo", "tasks": ["demo.run"],
                "weights": "weights.bin", "sha256": "0" * 64,
            }))
            manager = ModelArtifactManager(root)
            with self.assertRaises(ArtifactIntegrityError):
                manager.load_pytorch("model.json", model_factory=lambda: None)

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = ModelArtifactManager(root)
            with self.assertRaises(ArtifactIntegrityError):
                manager._inside_root("../outside/secret")

    def test_rejects_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ModelArtifactManager(directory)
            with self.assertRaises(ArtifactIntegrityError):
                manager._inside_root("/etc/passwd")


if __name__ == "__main__":
    unittest.main()
