import hashlib
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
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (Path(outside) / "secret").write_text("secret")
            (root / "manifest.json").write_text("{}")
            manager = ModelArtifactManager(root)
            with self.assertRaises((ArtifactIntegrityError, ValueError)):
                manager._inside_root("../" + str(Path(outside) / "secret").lstrip("/"))


if __name__ == "__main__":
    unittest.main()
