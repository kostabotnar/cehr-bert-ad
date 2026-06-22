"""Verify saved model/tokenizer artifacts and parse training result JSON.

Pure stdlib (json/os) — no torch/transformers needed, so this runs anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .reporting import StepReport

# A HF Trainer ``save_model`` writes config.json + a weights file + the tokenizer.
WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin"]
# CehrBertTokenizer.save_pretrained writes a tokenizer.json; accept common variants.
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "concept_tokenizer.json"]


def add_model_checks(report: StepReport, model_dir: Path, require_tokenizer: bool = True) -> StepReport:
    """Check that ``model_dir`` holds a usable, fully-saved model + tokenizer."""
    model_dir = Path(model_dir)
    if not report.add_check("model directory exists", model_dir.is_dir(), detail=str(model_dir)):
        return report

    has_config = (model_dir / "config.json").is_file()
    report.add_check("model config.json present", has_config)

    weights = [w for w in WEIGHT_FILES if (model_dir / w).is_file()]
    report.add_check("model weights present", bool(weights), detail=", ".join(weights) or f"none of {WEIGHT_FILES}")

    if require_tokenizer:
        tok = [t for t in TOKENIZER_FILES if (model_dir / t).is_file()]
        report.add_check("tokenizer files present", bool(tok), detail=", ".join(tok) or f"none of {TOKENIZER_FILES}")

    return report


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def add_train_results_checks(report: StepReport, model_dir: Path) -> StepReport:
    """Parse ``train_results.json`` / ``trainer_state.json`` and surface key metrics."""
    model_dir = Path(model_dir)

    train_results = _read_json(model_dir / "train_results.json")
    found = train_results is not None
    report.add_check("train_results.json present & valid", found, detail=str(model_dir / "train_results.json"))
    if found:
        loss = train_results.get("train_loss")
        report.add_metric("train_loss", loss)
        # A finite, non-negative loss is a cheap sanity signal that training ran.
        sane = isinstance(loss, (int, float)) and loss == loss and loss >= 0  # loss==loss rejects NaN
        report.add_check("train_loss is sane", sane, detail=f"train_loss={loss}", level="warning")
        if "epoch" in train_results:
            report.add_metric("epochs_completed", train_results["epoch"])

    state = _read_json(model_dir / "trainer_state.json")
    if state is not None:
        if "best_metric" in state:
            report.add_metric("best_metric", state["best_metric"])
        if "global_step" in state:
            report.add_metric("global_step", state["global_step"])
        if "epoch" in state and "epochs_completed" not in report.metrics:
            report.add_metric("epochs_completed", state["epoch"])
    else:
        report.warn("trainer_state.json present & valid", ok=False, detail="missing (optional)")

    return report
