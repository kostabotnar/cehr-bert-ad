import unittest

import numpy as np

from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import (
    ExtractTokenizedSequenceDataMapping,
    has_demographic_prompt,
)

DAY = 24 * 3600
INDEX = 1_000 * DAY


def make_record(person_id=1, with_prompt=True, first_visit_day=100, event_days=(300, 500, 700, 999)):
    """Sequence of [year, age, gender, race] (optional) then one token per event day."""
    ids, ages, epochs = [], [], []
    if with_prompt:
        ids += ["year:2015", "age:76", "8507", "8516"]
        ages += [-1] * 4
        epochs += [first_visit_day * DAY] * 4
    for d in event_days:
        ids.append(f"c{d}")
        ages.append(76)
        epochs.append(d * DAY)
    n = len(ids)
    return {
        "person_id": person_id,
        "concept_ids": ids,
        "input_ids": list(range(10, 10 + n)),
        "ages": ages,
        "epoch_times": epochs,
        "visit_concept_orders": list(range(n)),
        "concept_values": [0.0] * n,
        "num_of_visits": 3,
    }


def run(record, keep, window=365, index=INDEX):
    mapping = ExtractTokenizedSequenceDataMapping(
        {record["person_id"]: [{"index_date": index, "label": 1}]}, window, keep_demographic_prompt=keep
    )
    out = mapping.transform(record)
    return {k: v[0] for k, v in out.items()}


class TestKeepDemographicPrompt(unittest.TestCase):
    def test_prompt_kept_before_in_window_tokens_and_columns_aligned(self):
        record = make_record()
        out = run(record, keep=True)
        # Window is days 635 to 1000: events 700 and 999 are in, 300 and 500 are not.
        self.assertEqual(list(out["concept_ids"]), ["year:2015", "age:76", "8507", "8516", "c700", "c999"])
        self.assertEqual(list(out["input_ids"]), [10, 11, 12, 13, 16, 17])
        self.assertEqual(list(out["ages"]), [-1, -1, -1, -1, 76, 76])
        self.assertEqual(list(out["visit_concept_orders"]), [0, 1, 2, 3, 6, 7])
        self.assertEqual(list(out["epoch_times"])[:4], [100 * DAY] * 4)
        for column in ("concept_ids", "input_ids", "ages", "epoch_times", "visit_concept_orders", "concept_values"):
            self.assertEqual(len(out[column]), 6, column)
        self.assertEqual(out["num_of_visits"], 3)
        self.assertEqual(out["age_at_index"], 76)

    def test_option_off_keeps_old_behavior(self):
        out = run(make_record(), keep=False)
        self.assertEqual(list(out["concept_ids"]), ["c700", "c999"])

    def test_first_visit_inside_window_is_not_duplicated(self):
        record = make_record(first_visit_day=800, event_days=(850, 900))
        on = run(record, keep=True)
        off = run(record, keep=False)
        self.assertEqual(list(on["concept_ids"]), ["year:2015", "age:76", "8507", "8516", "c850", "c900"])
        self.assertEqual(list(on["concept_ids"]), list(off["concept_ids"]))

    def test_prompt_dated_after_window_end_is_dropped(self):
        record = make_record(first_visit_day=1200, event_days=(1300,))
        out = run(record, keep=True)
        self.assertEqual(len(out["concept_ids"]), 0)
        self.assertEqual(len(out["epoch_times"]), 0)

    def test_event_on_index_date_stays_excluded(self):
        record = make_record(event_days=(700, 1000))
        out = run(record, keep=True, index=INDEX - 1)
        self.assertEqual(list(out["concept_ids"])[4:], ["c700"])

    def test_record_without_prompt_is_unaffected(self):
        record = make_record(with_prompt=False)
        self.assertEqual(list(run(record, keep=True)["concept_ids"]), ["c700", "c999"])
        self.assertFalse(has_demographic_prompt(record["concept_ids"]))

    def test_unbounded_window_keeps_everything_once(self):
        record = make_record()
        out = run(record, keep=True, window=0)
        self.assertEqual(list(out["concept_ids"]), record["concept_ids"])

    def test_placeholder_gender_and_race_tokens_still_count_as_prompt(self):
        record = make_record()
        record["concept_ids"][2:4] = ["Gender/0", "Race/0"]
        out = run(record, keep=True)
        self.assertEqual(list(out["concept_ids"])[:4], ["year:2015", "age:76", "Gender/0", "Race/0"])

    def test_batch_transform_matches_transform(self):
        records = [make_record(1), make_record(2, with_prompt=False)]
        index_map = {r["person_id"]: [{"index_date": INDEX, "label": 0}] for r in records}
        mapping = ExtractTokenizedSequenceDataMapping(index_map, 365, keep_demographic_prompt=True)
        batch = {k: [r[k] for r in records] for k in records[0]}
        out = mapping.batch_transform(batch)
        self.assertEqual(len(out["concept_ids"][0]), 6)
        self.assertEqual(list(out["concept_ids"][1]), ["c700", "c999"])
        self.assertTrue(np.issubdtype(np.asarray(out["input_ids"][0]).dtype, np.integer))


if __name__ == "__main__":
    unittest.main()
