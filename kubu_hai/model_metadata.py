"""Safe loading and validation for optional model-metadata YAML files.

The loader validates descriptive metadata only. It never imports model code,
downloads artifacts, or executes values from a metadata document.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class ModelMetadataError(ValueError):
    """Raised when supplied model metadata is unreadable or invalid."""


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelMetadataError(f"{field} must be a non-empty string")
    return value.strip()


def validate_model_metadata(document: Any) -> dict[str, Any]:
    """Validate and return a normalized metadata mapping."""
    if not isinstance(document, Mapping):
        raise ModelMetadataError("model metadata must be a YAML mapping")

    metadata = dict(document)
    for field in ("name", "type", "targetType"):
        metadata[field] = _require_text(metadata.get(field), field)

    runtime_parameters = metadata.get("runtimeParameterDefinitions", [])
    if runtime_parameters is None:
        runtime_parameters = []
    if not isinstance(runtime_parameters, list):
        raise ModelMetadataError("runtimeParameterDefinitions must be a list")

    seen_fields: set[str] = set()
    normalized_parameters: list[dict[str, Any]] = []
    for index, parameter in enumerate(runtime_parameters):
        prefix = f"runtimeParameterDefinitions[{index}]"
        if not isinstance(parameter, Mapping):
            raise ModelMetadataError(f"{prefix} must be a mapping")
        item = dict(parameter)
        field_name = _require_text(item.get("fieldName"), f"{prefix}.fieldName")
        parameter_type = _require_text(item.get("type"), f"{prefix}.type")
        if field_name in seen_fields:
            raise ModelMetadataError(f"duplicate runtime parameter fieldName: {field_name}")
        seen_fields.add(field_name)
        item["fieldName"] = field_name
        item["type"] = parameter_type
        if "description" in item and item["description"] is not None:
            if not isinstance(item["description"], str):
                raise ModelMetadataError(f"{prefix}.description must be a string")
        normalized_parameters.append(item)
    metadata["runtimeParameterDefinitions"] = normalized_parameters

    target_type = metadata["targetType"].casefold()
    inference_model = metadata.get("inferenceModel")
    if target_type in {"binary", "multiclass"}:
        if not isinstance(inference_model, Mapping):
            raise ModelMetadataError(
                f"inferenceModel is required when targetType is {target_type}"
            )
        inference_model = dict(inference_model)
        inference_model["targetName"] = _require_text(
            inference_model.get("targetName"), "inferenceModel.targetName"
        )
        if target_type == "binary":
            inference_model["positiveClassLabel"] = _require_text(
                inference_model.get("positiveClassLabel"),
                "inferenceModel.positiveClassLabel",
            )
            inference_model["negativeClassLabel"] = _require_text(
                inference_model.get("negativeClassLabel"),
                "inferenceModel.negativeClassLabel",
            )
            if inference_model["positiveClassLabel"] == inference_model["negativeClassLabel"]:
                raise ModelMetadataError(
                    "binary positiveClassLabel and negativeClassLabel must differ"
                )
        else:
            labels = inference_model.get("classLabels")
            if (
                not isinstance(labels, list)
                or not labels
                or any(not isinstance(label, (str, int, float)) or isinstance(label, bool)
                       for label in labels)
            ):
                raise ModelMetadataError(
                    "inferenceModel.classLabels must be a non-empty list of scalar labels"
                )
            if len({str(label) for label in labels}) != len(labels):
                raise ModelMetadataError("inferenceModel.classLabels must be unique")
        metadata["inferenceModel"] = inference_model
    elif inference_model is not None and not isinstance(inference_model, Mapping):
        raise ModelMetadataError("inferenceModel must be a mapping when provided")

    return metadata


def load_model_metadata(path: str | Path | None) -> dict[str, Any] | None:
    """Load optional YAML metadata; absent path means metadata is not supplied.

    PyYAML is an optional dependency. A supplied but missing or malformed file
    is an error rather than being silently treated as absent.
    """
    if path is None:
        return None
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read model metadata; install kubu-hai-runtime[yaml]"
        ) from exc

    metadata_path = Path(path).expanduser()
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ModelMetadataError(f"could not read model metadata: {metadata_path}") from exc
    return validate_model_metadata(document)
