import os
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from datasets import DatasetDict, load_from_disk
from transformers.utils import logging

from cehrbert.data_generators.hf_data_generator.cache_util import CacheFileCollector
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import (
    ExtractTokenizedSequenceDataMapping,
    convert_index_date_to_utc_timestamp,
)
from cehrbert.runners.hf_runner_argument_dataclass import CehrBertArguments, DataTrainingArguments

LOG = logging.get_logger("transformers")


def _is_hf_dataset_dir(path: str) -> bool:
    """Return True if `path` is directly loadable by `load_from_disk`.

    A saved `Dataset` contains `dataset_info.json`/`state.json`, while a saved
    `DatasetDict` contains a top-level `dataset_dict.json`.
    """
    return os.path.isfile(os.path.join(path, "dataset_dict.json")) or os.path.isfile(
        os.path.join(path, "dataset_info.json")
    )


def resolve_tokenized_dataset_path(path: str) -> str:
    """Resolve the directory that pretraining actually saved the tokenized dataset to.

    The pretraining runner saves the prepared dataset under a hashed subdirectory
    (see `generate_prepared_ds_path`), so the configured `tokenized_full_dataset_path`
    typically points at the *base* folder, which is not itself a `Dataset`/`DatasetDict`
    directory. When that is the case, look for a single dataset subdirectory inside it
    and use that, so the config never needs to embed the (settings-dependent) hash.

    Args:
        path: The configured `tokenized_full_dataset_path`.

    Returns:
        A path that `load_from_disk` can load.

    Raises:
        FileNotFoundError: If more than one prepared dataset is found and the choice
            is ambiguous.
    """
    expanded = os.path.expanduser(path)

    # The configured path is already a loadable dataset directory.
    if _is_hf_dataset_dir(expanded):
        return expanded

    if os.path.isdir(expanded):
        dataset_subdirs = [
            os.path.join(expanded, name)
            for name in sorted(os.listdir(expanded))
            if _is_hf_dataset_dir(os.path.join(expanded, name))
        ]
        if len(dataset_subdirs) == 1:
            LOG.info("Resolved tokenized dataset path %s -> %s", expanded, dataset_subdirs[0])
            return dataset_subdirs[0]
        if len(dataset_subdirs) > 1:
            raise FileNotFoundError(
                f"Found multiple prepared datasets under {expanded}: {dataset_subdirs}. "
                f"Set tokenized_full_dataset_path to the specific subdirectory you want to use."
            )

    # Nothing better found; return the original so load_from_disk raises a clear error.
    return expanded


def extract_cohort_sequences(
    data_args: DataTrainingArguments,
    cehrbert_args: CehrBertArguments,
    cache_file_collector: Optional[CacheFileCollector] = None,
) -> DatasetDict:
    """
    Extracts and processes cohort-specific tokenized sequences from a pre-tokenized dataset,.

    based on the provided cohort Parquet files and observation window constraints.

    This function performs the following steps:
    1. Loads cohort definitions from Parquet files located in `data_args.cohort_folder`.
    2. Renames relevant columns if the data originates from a Meds format.
    3. Filters a pre-tokenized dataset (loaded from `cehrgpt_args.tokenized_full_dataset_path`)
       to include only patients present in the cohort.
    4. Aggregates each person's index date and label into a mapping.
    5. Checks for consistency to ensure all cohort person_ids are present in the tokenized dataset.
    6. Applies a transformation (`ExtractTokenizedSequenceDataMapping`) to generate
       observation-window-constrained patient sequences.
    7. Caches both the filtered and processed datasets using the provided `cache_file_collector`.

    Args:
        data_args (DataTrainingArguments): Configuration parameters for data processing,
            including cohort folder, observation window, batch size, and parallelism.
        cehrbert_args (CehrGPTArguments): Contains paths to pre-tokenized datasets and CEHR-GPT-specific arguments.
        cache_file_collector (CacheFileCollector): Utility to register and manage dataset cache files.

    Returns:
        DatasetDict: A Hugging Face `DatasetDict` containing the processed datasets (e.g., train/validation/test),
                     where each entry includes sequences filtered and truncated by the observation window.

    Raises:
        RuntimeError: If any `person_id` in the cohort is missing from the tokenized dataset.
    """

    cohort = pl.read_parquet(os.path.join(data_args.cohort_folder, "*.parquet"))
    if data_args.is_data_in_meds:
        cohort = cohort.rename(
            mapping={
                "prediction_time": "index_date",
                "subject_id": "person_id",
                "boolean_value": "label",
            }
        )
    all_person_ids = cohort["person_id"].unique().to_list()
    # In case the label column does not exist, we add a fake column to the dataframe so subsequent process can work
    if "label" not in cohort.columns:
        cohort = cohort.with_columns(pl.Series(name="label", values=np.zeros_like(cohort["person_id"].to_numpy())))

    # data_args.observation_window
    tokenized_dataset = load_from_disk(resolve_tokenized_dataset_path(cehrbert_args.tokenized_full_dataset_path))
    filtered_tokenized_dataset = tokenized_dataset.filter(
        lambda batch: [person_id in all_person_ids for person_id in batch["person_id"]],
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        num_proc=data_args.preprocessing_num_workers,
    )
    person_index_date_agg = cohort.group_by("person_id").agg(pl.struct("index_date", "label").alias("index_date_label"))
    # Convert to dictionary, normalizing index_date to a UTC POSIX timestamp (float) once here.
    # The cohort parquet may store index_date as either a date or a datetime; converting up front
    # keeps the downstream map() transform a simple float comparison and avoids repeating the
    # (multiprocess) conversion per record. See ExtractTokenizedSequenceDataMapping.transform.
    person_index_date_map: Dict[int, List[Dict[str, Any]]] = {
        person_id: [
            {
                "index_date": convert_index_date_to_utc_timestamp(entry["index_date"]),
                "label": entry["label"],
            }
            for entry in index_date_label
        ]
        for person_id, index_date_label in zip(
            person_index_date_agg["person_id"].to_list(),
            person_index_date_agg["index_date_label"].to_list(),
        )
    }
    LOG.info(f"person_index_date_agg: {person_index_date_agg}")
    tokenized_person_ids = []
    for _, dataset in filtered_tokenized_dataset.items():
        tokenized_person_ids.extend(dataset["person_id"])
    missing_person_ids = [
        person_id for person_id in person_index_date_map.keys() if person_id not in tokenized_person_ids
    ]
    if missing_person_ids:
        raise RuntimeError(
            f"There are {len(missing_person_ids)} missing in the tokenized dataset. "
            f"The list contains: {missing_person_ids}"
        )
    processed_dataset = filtered_tokenized_dataset.map(
        ExtractTokenizedSequenceDataMapping(person_index_date_map, data_args.observation_window).batch_transform,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=filtered_tokenized_dataset["train"].column_names,
    )
    return processed_dataset
