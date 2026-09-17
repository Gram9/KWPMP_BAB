"""
Leakage test: a CCM link's PERMNO must never be used for a date outside its
[linkdt, linkenddt] validity range. This is the leakage-relevant part of the
CRSP market-cap design (docs/superpowers/specs/2026-09-02-us-crsp-market-cap-design.md)
-- a gvkey can be reassigned to a different PERMNO over time, and using a
link valid only *after* as_of_date would let t+1 information influence a
decision made at t (CLAUDE.md's core rule).
"""

import pandas as pd

from src.data.market_cap import filter_valid_ccm_links


def test_link_valid_before_and_after_as_of_date_is_kept():
    links = pd.DataFrame({
        "gvkey": ["000001"],
        "lpermno": [10001],
        "linkdt": [pd.Timestamp("2010-01-01")],
        "linkenddt": [pd.Timestamp("2020-01-01")],
    })
    result = filter_valid_ccm_links(links, as_of_date=pd.Timestamp("2015-06-30"))
    assert len(result) == 1


def test_link_that_starts_after_as_of_date_is_excluded():
    """The core leakage case: this link only becomes valid in the future
    relative to as_of_date -- using it would be t+1 information at t."""
    links = pd.DataFrame({
        "gvkey": ["000001"],
        "lpermno": [10002],
        "linkdt": [pd.Timestamp("2015-07-01")],
        "linkenddt": [pd.Timestamp("2020-01-01")],
    })
    result = filter_valid_ccm_links(links, as_of_date=pd.Timestamp("2015-06-30"))
    assert result.empty


def test_link_that_ended_before_as_of_date_is_excluded():
    links = pd.DataFrame({
        "gvkey": ["000001"],
        "lpermno": [10003],
        "linkdt": [pd.Timestamp("2000-01-01")],
        "linkenddt": [pd.Timestamp("2010-01-01")],
    })
    result = filter_valid_ccm_links(links, as_of_date=pd.Timestamp("2015-06-30"))
    assert result.empty


def test_open_ended_link_null_linkenddt_is_kept_if_started():
    """linkenddt null means the link is still active (WRDS convention) --
    must be treated as open-ended, not accidentally excluded by a naive
    comparison against a null."""
    links = pd.DataFrame({
        "gvkey": ["000001"],
        "lpermno": [10004],
        "linkdt": [pd.Timestamp("2010-01-01")],
        "linkenddt": [pd.NaT],
    })
    result = filter_valid_ccm_links(links, as_of_date=pd.Timestamp("2025-01-01"))
    assert len(result) == 1


def test_gvkey_reassigned_permno_only_the_valid_one_survives():
    """Same gvkey, two sequential PERMNO links -- must not drop_duplicates
    blindly; the date bound must select the one link valid at as_of_date."""
    links = pd.DataFrame({
        "gvkey": ["000001", "000001"],
        "lpermno": [10005, 10006],
        "linkdt": [pd.Timestamp("2005-01-01"), pd.Timestamp("2012-01-01")],
        "linkenddt": [pd.Timestamp("2011-12-31"), pd.NaT],
    })
    result = filter_valid_ccm_links(links, as_of_date=pd.Timestamp("2015-06-30"))
    assert len(result) == 1
    assert result["lpermno"].iloc[0] == 10006
