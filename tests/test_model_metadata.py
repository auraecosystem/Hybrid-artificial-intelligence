import tempfile
import unittest
from pathlib import Path

from kubu_hai.model_metadata import (
    ModelMetadataError,
    load_model_metadata,
    validate_model_metadata,
)


class ModelMetadataTests(unittest.TestCase):
    def test_metadata_is_optional(self):
        self.assertIsNone(load_model_metadata(None))

    def test_valid_anomaly_metadata_and_runtime_parameters(self):
        result = validate_model_metadata({
            "name": "anomaly-model",
            "type": "inference",
            "targetType": "anomaly",
            "runtimeParameterDefinitions": [
                {"fieldName": "DEPLOYMENT_PARAMETER", "type": "deployment",
                 "description": "Deployment setting", "defaultValue": None},
                {"fieldName": "STRING_PARAMETER", "type": "string"},
            ],
        })
        self.assertEqual(result["name"], "anomaly-model")
        self.assertEqual(len(result["runtimeParameterDefinitions"]), 2)
        self.assertIsNone(result["runtimeParameterDefinitions"][0]["defaultValue"])

    def test_binary_requires_target_and_both_distinct_labels(self):
        valid = {
            "name": "binary-model", "type": "inference", "targetType": "binary",
            "inferenceModel": {
                "targetName": "outcome",
                "positiveClassLabel": "yes",
                "negativeClassLabel": "no",
            },
        }
        self.assertEqual(validate_model_metadata(valid)["inferenceModel"]["targetName"], "outcome")
        invalid = {**valid, "inferenceModel": {"targetName": "outcome"}}
        with self.assertRaises(ModelMetadataError):
            validate_model_metadata(invalid)

    def test_multiclass_requires_target_and_class_labels(self):
        valid = {
            "name": "multi-model", "type": "inference", "targetType": "multiclass",
            "inferenceModel": {"targetName": "species", "classLabels": ["cat", "dog"]},
        }
        self.assertEqual(len(validate_model_metadata(valid)["inferenceModel"]["classLabels"]), 2)
        invalid = {**valid, "inferenceModel": {"targetName": "species", "classLabels": []}}
        with self.assertRaises(ModelMetadataError):
            validate_model_metadata(invalid)

    def test_required_top_level_fields_are_enforced(self):
        for field in ("name", "type", "targetType"):
            document = {"name": "x", "type": "inference", "targetType": "anomaly"}
            document.pop(field)
            with self.subTest(field=field), self.assertRaises(ModelMetadataError):
                validate_model_metadata(document)

    def test_duplicate_runtime_parameter_names_are_rejected(self):
        document = {
            "name": "x", "type": "inference", "targetType": "anomaly",
            "runtimeParameterDefinitions": [
                {"fieldName": "MODE", "type": "string"},
                {"fieldName": "MODE", "type": "deployment"},
            ],
        }
        with self.assertRaises(ModelMetadataError):
            validate_model_metadata(document)

    def test_non_mapping_yaml_is_rejected(self):
        with self.assertRaises(ModelMetadataError):
            validate_model_metadata(["not", "a", "mapping"])

    def test_yaml_file_is_loaded(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model-metadata.yaml"
            path.write_text(
                "name: sample\ntype: inference\ntargetType: anomaly\n",
                encoding="utf-8",
            )
            self.assertEqual(load_model_metadata(path)["name"], "sample")


if __name__ == "__main__":
    unittest.main()
