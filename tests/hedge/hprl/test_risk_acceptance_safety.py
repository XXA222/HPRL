from collections.abc import Sequence

from freqtrade.hedge.hprl.risk_acceptance import (
    RiskLearningAcceptanceConfig,
    RiskLearningTrace,
    evaluate_risk_learning,
)


class _AmbiguousSequence(Sequence):
    """Sequence with NumPy/Torch-like invalid aggregate truth-value semantics."""

    def __init__(self, values):
        self._values = list(values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]

    def __bool__(self):
        raise ValueError("aggregate truth value is ambiguous")


def _trace(*, liquidations=0, ambiguous=False):
    levels = [0, 1, 2, 3, 2, 1, 0, 1] * 4
    margin = [0.0, 0.05, 0.12, 0.25, 0.12, 0.05, 0.0, 0.05] * 4
    returns = [0.0002, 0.0010, 0.0011, -0.0008, -0.0003, 0.0003, 0.0002, 0.0010] * 4
    drawdown = [0.0, 0.0, 0.0, 0.0008, 0.0011, 0.0009, 0.0006, 0.0001] * 4
    turnover = [0.0] * len(levels)
    projected = [False] * len(levels)
    if ambiguous:
        turnover = _AmbiguousSequence(turnover)
        projected = _AmbiguousSequence(projected)
    return RiskLearningTrace(
        equity_return=returns,
        gross_margin=margin,
        drawdown=drawdown,
        level_index=levels,
        turnover=turnover,
        projected=projected,
        liquidations=liquidations,
    )


def _config():
    return RiskLearningAcceptanceConfig(
        min_steps=16,
        min_distinct_levels=3,
        stress_drawdown_quantile=0.75,
        min_scale_in_success_rate=0.5,
        drawdown_penalty=0.2,
        cvar_penalty=1.0,
        turnover_penalty=0.0,
        require_baselines=False,
    )


def test_liquidation_blocks_otherwise_acceptable_policy():
    report = evaluate_risk_learning(_trace(liquidations=1), config=_config())
    assert report.verdict == "FAIL"
    assert report.realized_liquidations == 1
    assert "liquidation_observed" in report.reasons


def test_optional_sequences_do_not_require_aggregate_boolean_conversion():
    report = evaluate_risk_learning(_trace(ambiguous=True), config=_config())
    assert report.verdict == "PASS"
