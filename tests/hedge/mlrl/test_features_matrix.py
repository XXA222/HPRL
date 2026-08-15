from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.freqai.hedge_rl.features import (
    FeatureManifest,
    FeatureSchema,
    MissingValuePolicy,
    RobustFeatureScaler,
    StreamingMoments,
    apply_missing_value_policy,
    causal_rolling_zscore,
    evaluate_feature_freshness,
    greedy_correlation_prune,
    population_stability_index,
    sanitize_feature_matrix,
)


def test_round31_feature_schema_enforces_order_uniqueness_and_signature():
    schema = FeatureSchema(("a", "b"), "1.2.0")
    schema.validate_frame(pd.DataFrame({"a": [1], "b": [2]}))
    assert len(schema.signature) == 64 and schema.width == 2
    with pytest.raises(ValueError):
        FeatureSchema(("a", "a"))
    with pytest.raises(ValueError):
        schema.validate_frame(pd.DataFrame({"b": [2], "a": [1]}))


def test_round32_feature_sanitization_reports_replacements_and_clipping():
    matrix, report = sanitize_feature_matrix(
        [[np.nan, 99.0], [-99.0, 1.0]],
        clip=10,
        nonfinite_policy="zero",
    )
    assert np.isfinite(matrix).all()
    assert report.replaced_nonfinite == 1
    assert report.clipped_values == 2
    assert matrix.tolist() == [[0.0, 10.0], [-10.0, 1.0]]


def test_round33_robust_scaler_resists_large_outlier():
    values = np.array([[0.0], [1.0], [2.0], [1000.0]])
    scaler = RobustFeatureScaler().fit(values)
    transformed = scaler.transform(values)
    assert np.isfinite(transformed).all()
    assert scaler.median_[0] == 1.5
    assert transformed[0, 0] < transformed[2, 0] < transformed[3, 0]


def test_round34_streaming_moments_merge_matches_batch_statistics():
    values = np.arange(20, dtype=float).reshape(10, 2)
    left, right = StreamingMoments(2), StreamingMoments(2)
    left.update(values[:4])
    right.update(values[4:])
    left.merge(right)
    assert left.count == len(values)
    assert np.allclose(left.mean, values.mean(axis=0))
    assert np.allclose(left.variance, values.var(axis=0, ddof=1))


def test_round35_rolling_zscore_uses_only_prior_rows():
    values = np.array([[1.0], [2.0], [3.0], [4.0]])
    original = causal_rolling_zscore(values, window=3)
    changed = values.copy()
    changed[-1] = 4000.0
    changed_result = causal_rolling_zscore(changed, window=3)
    assert np.array_equal(original[:-1], changed_result[:-1])
    assert original[0, 0] == 0 and original[1, 0] == 0


def test_round36_missing_value_policies_are_bounded():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert apply_missing_value_policy(frame, policy=MissingValuePolicy.ZERO).iloc[1, 0] == 0
    result = apply_missing_value_policy(
        frame,
        policy=MissingValuePolicy.FORWARD_FILL,
        max_forward_fill=1,
    )
    assert result.iloc[1, 0] == 1
    with pytest.raises(ValueError):
        apply_missing_value_policy(
            pd.DataFrame({"a": [np.nan, 1.0]}),
            policy=MissingValuePolicy.FORWARD_FILL,
        )


def test_round37_feature_freshness_rejects_future_and_reports_age():
    result = evaluate_feature_freshness(produced_tick=10, decision_tick=12, max_age_steps=1)
    assert result.age_steps == 2 and not result.fresh and result.reason == "STALE_FEATURES"
    with pytest.raises(ValueError):
        evaluate_feature_freshness(produced_tick=13, decision_tick=12, max_age_steps=1)


def test_round38_population_stability_index_detects_distribution_shift():
    reference = np.linspace(-1, 1, 1000)
    same = population_stability_index(reference, reference.copy())
    shifted = population_stability_index(reference, reference + 2.0)
    assert same < 1e-12
    assert shifted > 0.5


def test_round39_correlation_prune_keeps_first_deterministically():
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [1, 1, 2, 3]})
    kept = greedy_correlation_prune(frame, threshold=0.99)
    assert kept == ("a", "c")


def test_round40_feature_manifest_changes_with_data_and_has_stats():
    schema = FeatureSchema(("a", "b"))
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 5.0]})
    first = FeatureManifest.build(frame, schema)
    second = FeatureManifest.build(frame.assign(b=[3.0, 6.0]), schema)
    assert first.row_count == 2
    assert first.means == (1.5, 4.0)
    assert first.fingerprint != second.fingerprint
