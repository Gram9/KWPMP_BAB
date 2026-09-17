"""Tests for src/data/risk_free_canada.py -- Canadian risk-free rate,
per config/risk_free.yaml's `canada` block.

CHASS's monthly parquet is a per-security panel (ind2/ind1 repeat on
every row for a given month), so the loader must dedupe to one row per
month -- and must emit rf_source distinguishing real ind2 observations
from the two derived_overrides months (2023-10-31, 2023-11-30), which
must never be indistinguishable from observed data downstream.
"""

import datetime

import polars as pl
import pytest

from src.data import risk_free_canada

_CHASS_COLUMN = "ind2-30 day Return on T-Bills"
_DATE_COLUMN = "trdate-Trade Date"


@pytest.fixture
def fixture_parquet(tmp_path, monkeypatch):
    # Mimics the real file's shape: multiple security rows per month,
    # same ind2 value repeated -- the loader must dedupe by date, not
    # just read the column as-is (which would multiply-count months).
    # Suppression reason: real CHASS parquet dates are naive datetimes
    # (Datetime(time_unit='ns', time_zone=None), verified against the
    # real file) -- matching that here is intentional, not an oversight.
    df = pl.DataFrame(
        {
            _DATE_COLUMN: [
                datetime.datetime(1980, 1, 31),  # noqa: DTZ001
                datetime.datetime(1980, 1, 31),  # noqa: DTZ001
                datetime.datetime(1980, 1, 31),  # noqa: DTZ001
                datetime.datetime(1980, 2, 29),  # noqa: DTZ001
                datetime.datetime(1980, 2, 29),  # noqa: DTZ001
                datetime.datetime(2023, 10, 31),  # noqa: DTZ001
                datetime.datetime(2023, 11, 30),  # noqa: DTZ001
            ],
            _CHASS_COLUMN: [0.011241, 0.011241, 0.011241, 0.010768, 0.010768, None, None],
        }
    )
    path = tmp_path / "monthly_fixture.parquet"
    df.write_parquet(path)
    monkeypatch.setattr(risk_free_canada, "CHASS_MONTHLY_PATH", path)
    return path


def test_load_canada_rf_chass_dedupes_to_one_row_per_month(fixture_parquet):
    result = risk_free_canada.load_canada_rf_chass()

    assert result.height == result["date"].n_unique()


def test_load_canada_rf_chass_real_values_are_decimal_not_percent(fixture_parquet):
    result = risk_free_canada.load_canada_rf_chass()

    row = result.filter(pl.col("date") == pl.date(1980, 1, 31))
    assert row.height == 1
    # 0.011241 is already decimal -- must not be scaled by /100 or *100.
    assert row["rf"][0] == pytest.approx(0.011241, rel=1e-9)


def test_load_canada_rf_chass_flags_real_observations_as_chass_source(fixture_parquet):
    result = risk_free_canada.load_canada_rf_chass()

    row = result.filter(pl.col("date") == pl.date(1980, 1, 31))
    assert row["rf_source"][0] == "chass"


def test_load_canada_rf_chass_derived_months_carry_non_chass_source(fixture_parquet):
    result = risk_free_canada.load_canada_rf_chass()

    oct_row = result.filter(pl.col("date") == pl.date(2023, 10, 31))
    nov_row = result.filter(pl.col("date") == pl.date(2023, 11, 30))
    assert oct_row.height == 1
    assert nov_row.height == 1
    # These two months must NEVER pass as observed ind2 data -- per
    # config/risk_free.yaml's derived_overrides block.
    assert oct_row["rf_source"][0] == "derived_from_external_rate"
    assert nov_row["rf_source"][0] == "derived_from_ind1"
    assert oct_row["rf"][0] == pytest.approx(0.00410833, rel=1e-6)
    assert nov_row["rf"][0] == pytest.approx(0.00420167, rel=1e-6)


def test_load_canada_rf_chass_no_row_has_null_rf(fixture_parquet):
    # The whole point of derived_overrides is that these two months are
    # filled in, not left null to vanish silently in a downstream dropna.
    result = risk_free_canada.load_canada_rf_chass()

    assert result["rf"].null_count() == 0


def test_load_canada_rf_chass_real_file_dedupes_to_552_months():
    result = risk_free_canada.load_canada_rf_chass()

    # config/risk_free.yaml: CHASS coverage 1980-01-31 to 2025-12-31,
    # 552 months, exactly matching the CHASS equity panel's own range.
    assert result.height == 552
    assert result["rf"].null_count() == 0


def test_load_canada_rf_chass_real_file_only_two_months_are_non_chass():
    result = risk_free_canada.load_canada_rf_chass()

    non_chass = result.filter(pl.col("rf_source") != "chass")
    assert non_chass.height == 2
    assert set(non_chass["date"].to_list()) == {
        datetime.date(2023, 10, 31),
        datetime.date(2023, 11, 30),
    }


def test_load_canada_rf_chass_raises_if_month_has_disagreeing_values(tmp_path, monkeypatch):
    # leakage-auditor finding: plain unique() gives no guarantee about
    # which duplicate survives when security rows disagree on ind2 for
    # the same month. Must fail loudly, not silently pick one.
    df = pl.DataFrame(
        {
            _DATE_COLUMN: [
                datetime.datetime(1980, 1, 31),  # noqa: DTZ001
                datetime.datetime(1980, 1, 31),  # noqa: DTZ001
            ],
            _CHASS_COLUMN: [0.011241, 0.099999],
        }
    )
    path = tmp_path / "disagreeing_fixture.parquet"
    df.write_parquet(path)
    monkeypatch.setattr(risk_free_canada, "CHASS_MONTHLY_PATH", path)

    with pytest.raises(AssertionError, match="distinct"):
        risk_free_canada.load_canada_rf_chass()


def test_load_canada_rf_chass_raises_if_override_month_has_a_real_observation(
    tmp_path, monkeypatch
):
    # leakage-auditor finding: a mixed null/non-null month (some security
    # rows null, one carrying a real ind2 value) must not silently dedup
    # to a null row and let the override overwrite a genuine observation
    # with the hardcoded proxy value.
    df = pl.DataFrame(
        {
            _DATE_COLUMN: [
                datetime.datetime(2023, 10, 31),  # noqa: DTZ001
                datetime.datetime(2023, 10, 31),  # noqa: DTZ001
                datetime.datetime(2023, 10, 31),  # noqa: DTZ001
                datetime.datetime(2023, 11, 30),  # noqa: DTZ001
            ],
            _CHASS_COLUMN: [None, None, 0.00999, None],
        }
    )
    path = tmp_path / "mixed_null_fixture.parquet"
    df.write_parquet(path)
    monkeypatch.setattr(risk_free_canada, "CHASS_MONTHLY_PATH", path)

    # The disagreeing-values guard fires first (None vs 0.00999 are two
    # distinct values), which is also a correct refusal for this input.
    with pytest.raises(AssertionError, match="distinct"):
        risk_free_canada.load_canada_rf_chass()
