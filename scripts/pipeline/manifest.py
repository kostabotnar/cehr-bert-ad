"""Prediction manifests: how a model part tells the evaluation part where its predictions are.

A manifest is ``predictions_manifest.json`` with the shape::

    {"models": [{"name", "display_label", "family", "test_predictions",
                 "validation_predictions", "secondary"}, ...]}

Prediction paths are stored relative to ``PROJECT_ROOT`` when the folder is inside the project and as
absolute paths otherwise (``to_manifest_path``); ``read_manifest`` resolves either form to an absolute
path. Each prediction folder holds parquet files in the CEHR-BERT schema
(subject_id, prediction_time, predicted_boolean_probability, boolean_value).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from . import paths

REQUIRED_KEYS = ("name", "display_label", "family", "test_predictions", "validation_predictions", "secondary")


class ManifestError(ValueError):
    """The manifest file is unreadable or violates the contract."""


@dataclasses.dataclass(frozen=True)
class ManifestModel:
    name: str
    display_label: str
    family: str
    test_predictions: Path
    validation_predictions: Path
    secondary: bool = False


def to_manifest_path(path: str | Path) -> str:
    """Project-relative posix path when ``path`` is under PROJECT_ROOT, else the absolute path."""
    p = Path(path)
    if not p.is_absolute():
        p = paths.PROJECT_ROOT / p
    try:
        return p.relative_to(paths.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


def make_entry(
    name: str,
    display_label: str,
    family: str,
    test_predictions: str | Path,
    validation_predictions: str | Path,
    secondary: bool = False,
) -> dict:
    return {
        "name": name,
        "display_label": display_label,
        "family": family,
        "test_predictions": to_manifest_path(test_predictions),
        "validation_predictions": to_manifest_path(validation_predictions),
        "secondary": bool(secondary),
    }


def _validate(data, source: str) -> list[dict]:
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ManifestError(f"{source}: expected an object with a 'models' list")
    seen: set[str] = set()
    for i, entry in enumerate(data["models"]):
        if not isinstance(entry, dict):
            raise ManifestError(f"{source}: models[{i}] is not an object")
        missing = [k for k in REQUIRED_KEYS if k not in entry]
        if missing:
            raise ManifestError(f"{source}: models[{i}] is missing keys {missing}")
        if not isinstance(entry["name"], str) or not entry["name"]:
            raise ManifestError(f"{source}: models[{i}] has an empty name")
        if entry["name"] in seen:
            raise ManifestError(f"{source}: duplicate model name {entry['name']!r}")
        if not isinstance(entry["secondary"], bool):
            raise ManifestError(f"{source}: models[{i}] 'secondary' must be true or false")
        seen.add(entry["name"])
    return data["models"]


def read_manifest(path: str | Path) -> list[ManifestModel]:
    """Read and validate a manifest; prediction paths come back absolute."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{path}: cannot read manifest ({exc})") from exc
    return [
        ManifestModel(
            name=e["name"],
            display_label=e["display_label"],
            family=e["family"],
            test_predictions=paths.resolve_under_root(e["test_predictions"]),
            validation_predictions=paths.resolve_under_root(e["validation_predictions"]),
            secondary=e["secondary"],
        )
        for e in _validate(data, str(path))
    ]


def upsert_models(manifest_path: str | Path, entries: list[dict]) -> Path:
    """Replace entries with the same name and keep the others; creates the file when absent."""
    manifest_path = Path(manifest_path)
    existing: list[dict] = []
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = _validate(data, str(manifest_path))
    new = {e["name"]: e for e in entries}
    merged = [new.pop(e["name"], e) for e in existing] + list(new.values())
    _validate({"models": merged}, str(manifest_path))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"models": merged}, indent=2), encoding="utf-8")
    return manifest_path


def add_selection_args(parser) -> None:
    """The options evaluation steps 01 and 02 share for choosing which manifests (and so models) to use."""
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        metavar="PATH",
        help="Predictions manifest to include (repeatable). Default: the cehrbert_model manifest for "
        "--window-days and --ctx and the classical_ml manifest for --window-days.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=paths.DEFAULT_WINDOW_DAYS,
        help="Look-back window in days for the default manifests (negative = all history).",
    )
    parser.add_argument("--ctx", type=int, default=paths.DEFAULT_CTX, help="Token context for the default manifests.")
    parser.add_argument(
        "--name",
        default=None,
        help="Comparison name, the folder under build/evaluation/ (default: the cehrbert run label).",
    )


def comparison_name(args) -> str:
    return args.name or paths.run_label(args.window_days, args.ctx)


def collect_models(args) -> tuple[list[ManifestModel], list[tuple[bool, str, str]]]:
    """Models of the selected manifests plus findings as ``(is_failure, check name, detail)``.

    An explicit ``--manifest`` that cannot be read is a failure. A missing default manifest only warns, and the
    family is skipped. Model names must be unique across all manifests.
    """
    findings: list[tuple[bool, str, str]] = []
    if args.manifest:
        sources = [(Path(m), True, f"manifest {m}") for m in args.manifest]
    else:
        label = paths.run_label(args.window_days, args.ctx)
        sources = [
            (paths.cehrbert_manifest_path(label), False, "cehrbert_model manifest"),
            (paths.classical_manifest_path(paths.window_label(args.window_days)), False, "classical_ml manifest"),
        ]
    models: list[ManifestModel] = []
    seen: dict[str, Path] = {}
    for path, explicit, what in sources:
        check = f"{what} readable"
        if not path.is_file() and not explicit:
            findings.append((False, check, f"no manifest at {path}; that model family is skipped"))
            continue
        try:
            found = read_manifest(path)
        except ManifestError as exc:
            findings.append((explicit, check, str(exc)))
            continue
        for model in found:
            if model.name in seen:
                findings.append((True, f"model name {model.name!r} unique", f"in both {seen[model.name]} and {path}"))
                continue
            seen[model.name] = path
            models.append(model)
    return models, findings


def primary_first(models: list[ManifestModel]) -> list[ManifestModel]:
    """Stable order in which, within a family, primary models precede secondary ones; families keep their order."""
    family_order: dict[str, int] = {}
    for m in models:
        family_order.setdefault(m.family, len(family_order))
    return sorted(models, key=lambda m: (family_order[m.family], m.secondary))
