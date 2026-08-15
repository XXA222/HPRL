from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.freqai.hedge_rl.market_data import (
    align_funding_rates,
    dataset_fingerprint,
    detect_market_gaps,
    deterministic_episode_starts,
    duplicate_timestamps,
    purged_chronological_split,
    purged_walk_forward,
    stationary_block_bootstrap_indices,
    validate_chronology,
    validate_ohlc_consistency,
)
from tests.hedge.mlrl.advanced_helpers import synthetic_features, synthetic_prices


def test_round41_chronology_requires_sorted_unique_timezone_aware_index():
    index = pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC")
    validate_chronology(index)
    with pytest.raises(ValueError):
        validate_chronology(index[::-1])
    with pytest.raises(ValueError):
        validate_chronology(pd.DatetimeIndex([index[0], index[0]]))


def test_round42_duplicate_timestamp_detection_is_deduplicated():
    index = pd.Index([1, 2, 2, 3, 3, 3])
    assert duplicate_timestamps(index) == (2, 3)


def test_round43_ohlc_consistency_reports_exact_invalid_rows():
    prices = synthetic_prices(8).reset_index(drop=True)
    prices.loc[3, "high"] = prices.loc[3, "low"] - 1
    report = validate_ohlc_consistency(prices, raise_on_error=False)
    assert not report.valid and report.invalid_rows == (3,)
    with pytest.raises(ValueError):
        validate_ohlc_consistency(prices)


def test_round44_market_gap_detector_counts_missing_intervals():
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-01T00:00:00Z"),
            pd.Timestamp("2026-01-01T00:01:00Z"),
            pd.Timestamp("2026-01-01T00:04:00Z"),
        ]
    )
    gaps = detect_market_gaps(index, expected_interval=pd.Timedelta(minutes=1))
    assert len(gaps) == 1 and gaps[0].missing_intervals == 2


def test_round45_funding_alignment_is_backward_only():
    candles = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    funding = pd.Series([0.001, 0.002], index=[candles[1], candles[3]])
    aligned = align_funding_rates(candles, funding)
    assert aligned.tolist() == [0.0, 0.001, 0.001, 0.002, 0.002]


def test_round46_purged_split_has_embargo_and_no_overlap():
    split = purged_chronological_split(100, embargo=3)
    train, validation, test = split.indexes()
    assert not (train & validation or train & test or validation & test)
    assert split.validation.start - split.train.stop == 3
    assert split.test.start - split.validation.stop == 3


def test_round47_walk_forward_folds_are_purged_and_ordered():
    folds = purged_walk_forward(100, train_length=30, evaluation_length=10, embargo=2, step=10)
    assert len(folds) == 6
    assert all(fold.evaluation.start - fold.train.stop == 2 for fold in folds)
    assert tuple(fold.fold for fold in folds) == tuple(range(len(folds)))


def test_round48_episode_start_sampling_is_deterministic_and_in_bounds():
    first = deterministic_episode_starts(
        region_start=0,
        region_stop=100,
        window=8,
        episode_steps=16,
        count=5,
        seed=602,
    )
    second = deterministic_episode_starts(
        region_start=0,
        region_stop=100,
        window=8,
        episode_steps=16,
        count=5,
        seed=602,
    )
    assert first == second and len(set(first)) == 5
    assert all(7 <= item <= 83 for item in first)


def test_round49_block_bootstrap_preserves_contiguous_blocks_circularly():
    indices = stationary_block_bootstrap_indices(10, block_length=3, output_length=9, seed=602)
    assert indices.shape == (9,) and indices.dtype == np.int64
    for start in range(0, 9, 3):
        block = indices[start : start + 3]
        assert np.array_equal((block - block[0]) % 10, np.array([0, 1, 2]))


def test_round50_dataset_fingerprint_is_content_and_metadata_sensitive():
    prices = synthetic_prices(12)
    features = synthetic_features(prices)
    first = dataset_fingerprint(features, prices, metadata={"version": 1})
    assert first == dataset_fingerprint(features.copy(), prices.copy(), metadata={"version": 1})
    changed = features.copy()
    changed.iloc[0, 0] += 1
    assert first != dataset_fingerprint(changed, prices, metadata={"version": 1})
    assert first != dataset_fingerprint(features, prices, metadata={"version": 2})
