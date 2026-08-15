from decimal import Decimal

from freqtrade.hedge.readiness import (
    ReadinessGate,
    ReadinessInputs,
    ReadinessReasonCode,
    ReadinessState,
)


def ready_inputs(**overrides):
    values = dict(
        database_migration_succeeded=True,
        single_writer_lease_valid=True,
        position_mode="hedge",
        margin_mode="cross",
        configured_leverage=Decimal("3"),
        observed_leverages=(Decimal("3"),),
        unmanaged_position_count=0,
        unmanaged_order_count=0,
        rest_snapshot_valid=True,
        user_stream_fresh=True,
        unknown_order_count=0,
        reconciliation_converged=True,
        risk_data_valid=True,
        halt_reasons=(),
    )
    values.update(overrides)
    return ReadinessInputs(**values)


def test_ready_requires_every_condition() -> None:
    report = ReadinessGate(clock_ms=lambda: 1).evaluate(ready_inputs())
    assert report.state is ReadinessState.READY
    assert not report.reason_codes


def test_every_failure_has_stable_reason_code() -> None:
    report = ReadinessGate(clock_ms=lambda: 1).evaluate(
        ready_inputs(
            database_migration_succeeded=False,
            single_writer_lease_valid=False,
            position_mode="oneway",
            margin_mode="isolated",
            observed_leverages=(Decimal("5"),),
            unmanaged_position_count=1,
            unmanaged_order_count=1,
            rest_snapshot_valid=False,
            user_stream_fresh=False,
            unknown_order_count=1,
            reconciliation_converged=False,
            risk_data_valid=False,
            halt_reasons=("MANUAL_HALT",),
        )
    )
    assert report.state is ReadinessState.HALT
    assert len(report.reason_codes) == 16
    assert all(isinstance(code, ReadinessReasonCode) for code in report.reason_codes)


def test_stale_stream_blocks_new_risk_but_allows_controlled_reduce() -> None:
    gate = ReadinessGate(clock_ms=lambda: 1)
    report = gate.evaluate(ready_inputs(user_stream_fresh=False))
    assert report.state is ReadinessState.DEGRADED
    assert not gate.allows_new_risk()
    assert gate.allows_controlled_reduce()


def test_readiness_counts_must_be_integers() -> None:
    import pytest

    with pytest.raises(ValueError, match="nonnegative integer"):
        ready_inputs(unmanaged_order_count=0.5)


def test_readiness_rejects_non_boolean_flags() -> None:
    import pytest

    with pytest.raises(ValueError, match="rest_snapshot_valid must be a boolean"):
        ready_inputs(rest_snapshot_valid="false")


def test_readiness_rejects_string_halt_reasons() -> None:
    import pytest

    with pytest.raises(ValueError, match="halt_reasons must be a tuple"):
        ready_inputs(halt_reasons="MANUAL_HALT")


def test_invalid_clock_replaces_old_ready_state_with_stable_failure() -> None:
    now = [1]
    gate = ReadinessGate(clock_ms=lambda: now[0])
    assert gate.evaluate(ready_inputs()).ready
    now[0] = -1
    report = gate.evaluate(ready_inputs())
    assert report.state is ReadinessState.NOT_READY
    assert report.reason_codes == (ReadinessReasonCode.READINESS_CLOCK_INVALID,)
    assert not gate.allows_new_risk()
    assert not gate.allows_controlled_reduce()


def test_leverage_below_one_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="greater than or equal to 1"):
        ready_inputs(configured_leverage=Decimal("0.5"))
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        ready_inputs(observed_leverages=(Decimal("0.5"),))


def test_raw_snapshot_age_is_evaluated_by_gate() -> None:
    gate = ReadinessGate(clock_ms=lambda: 2000)
    report = gate.evaluate(
        ready_inputs(
            rest_snapshot_observed_at_ms=1000,
            rest_snapshot_max_age_ms=500,
        )
    )
    assert report.state is ReadinessState.DEGRADED
    assert ReadinessReasonCode.REST_SNAPSHOT_STALE in report.reason_codes
