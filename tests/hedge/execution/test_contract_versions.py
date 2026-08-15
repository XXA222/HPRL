from freqtrade.hedge.contracts import (
    CONTRACTS_VERSION,
    EVENT_VERSION,
    PAYLOAD_VERSION,
    HedgeContractError,
    ReasonCode,
)


def test_contract_versions_and_reason_codes_are_stable():
    assert CONTRACTS_VERSION == "2.0"
    assert EVENT_VERSION == "1.0"
    assert PAYLOAD_VERSION == "1.0"
    error = HedgeContractError(ReasonCode.UNKNOWN_ORDER)
    assert error.reason_code is ReasonCode.UNKNOWN_ORDER
    assert str(error) == "UNKNOWN_ORDER"
