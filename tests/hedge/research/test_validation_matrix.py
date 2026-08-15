"""One independently addressable pytest node for every research validation check."""

from __future__ import annotations

from freqtrade.hedge.research.validation_matrix import (
    ROUND_SPECS,
    validate_registry,
    validate_round,
)


def test_round_registry_is_exact() -> None:
    validate_registry()


def _make_round_test(round_no: int):
    def test_round() -> None:
        spec = validate_round(round_no)
        assert spec.round_no == round_no

    test_round.__name__ = f"test_round_{round_no:03d}"
    return test_round


for _spec in ROUND_SPECS:
    globals()[f"test_round_{_spec.round_no:03d}"] = _make_round_test(_spec.round_no)
