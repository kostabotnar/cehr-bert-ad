"""Checks for HuggingFace ``DatasetDict`` artifacts (prepared/tokenized datasets).

``datasets`` is imported lazily so this module imports cleanly without it; functions
that need it raise a clear error or record a skipped check instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from .reporting import StepReport


def datasets_available() -> bool:
    try:
        import datasets  # noqa: F401

        return True
    except Exception:
        return False


def _is_hf_dataset_dir(path: str) -> bool:
    """True if ``path`` is directly loadable by ``load_from_disk``.

    Mirrors cehrbert.runners.data_utils._is_hf_dataset_dir without importing cehrbert.
    """
    return os.path.isfile(os.path.join(path, "dataset_dict.json")) or os.path.isfile(
        os.path.join(path, "dataset_info.json")
    )


def resolve_prepared_path(path: str | Path) -> Optional[str]:
    """Resolve the directory that actually holds the saved dataset.

    Pretraining saves under a hashed subdirectory, so the configured base folder is
    usually not itself loadable. Returns the loadable directory, or None if it can't
    be resolved unambiguously.
    """
    expanded = os.path.expanduser(str(path))
    if _is_hf_dataset_dir(expanded):
        return expanded
    if os.path.isdir(expanded):
        subdirs = [
            os.path.join(expanded, name)
            for name in sorted(os.listdir(expanded))
            if _is_hf_dataset_dir(os.path.join(expanded, name))
        ]
        if len(subdirs) == 1:
            return subdirs[0]
        if len(subdirs) > 1:
            return None  # ambiguous
    return None


def add_prepared_dataset_checks(
    report: StepReport,
    prepared_dir: Path,
    expected_splits: Optional[List[str]] = None,
    require_person_id: bool = True,
) -> StepReport:
    """Check that a prepared/tokenized DatasetDict exists, is loadable, has splits.

    If ``datasets`` is unavailable, presence is checked structurally and the load is
    recorded as a skipped (warning) check rather than failing the step.
    """
    prepared_dir = Path(prepared_dir)
    if not report.add_check("prepared dataset dir exists", prepared_dir.is_dir(), detail=str(prepared_dir)):
        return report

    resolved = resolve_prepared_path(prepared_dir)
    if not report.add_check("prepared dataset is loadable directory", resolved is not None,
                            detail=resolved or "no single DatasetDict/Dataset found (or ambiguous)"):
        return report

    if not datasets_available():
        report.warn("dataset contents verified", ok=False,
                    detail="`datasets` not installed; structural check only")
        return report

    from datasets import DatasetDict, load_from_disk

    try:
        dataset = load_from_disk(resolved)
    except Exception as exc:
        report.add_check("dataset loads via load_from_disk", False, detail=str(exc))
        return report
    report.add_check("dataset loads via load_from_disk", True)

    if isinstance(dataset, DatasetDict):
        splits = list(dataset.keys())
        report.add_metric("splits", ",".join(splits))
        for name in splits:
            report.add_metric(f"{name}_rows", dataset[name].num_rows)
        if expected_splits:
            missing = [s for s in expected_splits if s not in splits]
            report.add_check("expected splits present", not missing,
                             detail="ok" if not missing else f"missing: {', '.join(missing)}")
        sample_split = dataset[splits[0]]
        columns = sample_split.column_names
    else:  # a single Dataset
        report.add_metric("rows", dataset.num_rows)
        columns = dataset.column_names

    if require_person_id:
        report.add_check("dataset has person_id column", "person_id" in columns,
                         detail=f"columns: {', '.join(columns[:12])}{'...' if len(columns) > 12 else ''}")

    return report


def count_person_ids(prepared_dir: Path) -> set:
    """Collect the set of person_ids across all splits (requires ``datasets``)."""
    if not datasets_available():
        raise RuntimeError("`datasets` is required to count person_ids")
    from datasets import load_from_disk

    resolved = resolve_prepared_path(prepared_dir)
    if resolved is None:
        raise FileNotFoundError(f"No loadable dataset under {prepared_dir}")
    dataset = load_from_disk(resolved)
    ids: set = set()
    if hasattr(dataset, "values"):
        for split in dataset.values():
            ids.update(split["person_id"])
    else:
        ids.update(dataset["person_id"])
    return ids


def collapse_to_test_split(prepared_root: Path) -> List[str]:
    """Collapse all splits of cached cohort DatasetDict(s) into a single ``test`` split.

    The cehrbert runner's extract_cohort_sequences route builds train/validation splits;
    a predict-only run needs ``processed_dataset["test"]``. This pools all splits into one
    ``test`` split so the (possibly unmodified) runner can predict on it. Idempotent: a
    DatasetDict that is already exactly ``{"test"}`` is left untouched. Resilient: a broken
    sibling dir is skipped. Returns the dataset directory names that were collapsed.
    """
    if not datasets_available():
        return []
    import shutil

    from datasets import DatasetDict, concatenate_datasets, load_from_disk

    patched: List[str] = []
    for marker in sorted(Path(prepared_root).glob("*/dataset_dict.json")):
        ds_dir = marker.parent
        try:
            ds = load_from_disk(str(ds_dir))
            if not isinstance(ds, DatasetDict) or set(ds.keys()) == {"test"}:
                continue
            combined = concatenate_datasets([ds[k] for k in sorted(ds.keys())])
            tmp_dir = ds_dir.with_name(ds_dir.name + "_test")
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            DatasetDict({"test": combined}).save_to_disk(str(tmp_dir))
            shutil.rmtree(ds_dir)
            tmp_dir.rename(ds_dir)
            patched.append(ds_dir.name)
        except Exception:  # noqa: BLE001 - resilience: skip a bad dir, keep going
            continue
    return patched

