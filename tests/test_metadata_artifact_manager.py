import tempfile
import unittest
from pathlib import Path

from kubu_hai.artifacts import ArtifactIntegrityError, ModelArtifactManager


class ArtifactMetadataTests(unittest.TestCase):
    def test_metadata_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ModelArtifactManager(directory)
            self.assertIsNone(manager.load_metadata())

    def test_metadata_loads_from_artifact_root(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            path.write_text(
                "name: example\ntype: inference\ntargetType: anomaly\n",
                encoding="utf-8",
            )
            manager = ModelArtifactManager(directory)
            self.assertEqual(manager.load_metadata("model.yaml")["name"], "example")

    def test_metadata_cannot_escape_artifact_root(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ModelArtifactManager(directory)
            with self.assertRaises(ArtifactIntegrityError):
                manager.load_metadata("../outside.yaml")


if __name__ == "__main__":
    unittest.main()
