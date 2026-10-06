import polars as pl
import pytest


@pytest.mark.parametrize(
    "value,expected",
    [
        ("ICD-10-CM:G30.9", True),
        ("ICD-10-CM:G30", True),
        ("ICD-9-CM:331.0", True),
        ("ICD-10-CM:G31.0", False),
        ("ICD-9-CM:331.1", False),
        ("X-ICD-10-CM:G30", False),
        (None, None),
    ],
)
def test_ad_source_value_is_prefix_match(step, value, expected):
    mod = step("06_prepare_finetuning_data.py")
    df = pl.DataFrame({"condition_source_value": [value]}, schema={"condition_source_value": pl.String})
    assert df.select(mod.is_ad_source_value()).item() == expected
