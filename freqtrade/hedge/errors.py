"""Canonical Hedge exception hierarchy with stable reason codes.

The original exception names remain source compatible.  New integration boundaries
use the specialized persistence/projection/contract/retry classes below so callers
can make explicit fail-closed and retry decisions without parsing error messages.
"""

from __future__ import annotations


class HedgeError(RuntimeError, ValueError):
    """Base class for hedge-domain failures."""

    default_reason_code = "HEDGE_ERROR"

    def __init__(self, message: str = "Hedge operation failed.", *, reason_code: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code or self.default_reason_code


class HedgeConfigurationError(HedgeError):
    """Raised when a hedge configuration or domain value is invalid."""

    default_reason_code = "HEDGE_CONFIGURATION_ERROR"


class HedgeDataError(HedgeConfigurationError):
    """Malformed, non-finite, stale, or contradictory external data."""

    default_reason_code = "HEDGE_DATA_ERROR"


class HedgeStateError(HedgeError):
    """Raised when an operation is invalid for the current hedge state."""

    default_reason_code = "HEDGE_STATE_ERROR"


class HedgeInvariantError(HedgeStateError):
    """A domain invariant was violated."""

    default_reason_code = "HEDGE_INVARIANT_ERROR"


class HedgeSafetyError(HedgeStateError):
    """A fail-closed condition requires DEGRADED or HALT."""

    default_reason_code = "HEDGE_SAFETY_ERROR"


class HedgeIntegrationError(HedgeError):
    """Raised when two hedge subsystems cannot be composed."""

    default_reason_code = "HEDGE_INTEGRATION_ERROR"


class HedgePersistenceError(HedgeIntegrationError):
    default_reason_code = "HEDGE_PERSISTENCE_ERROR"


class HedgeProjectionError(HedgeIntegrationError):
    default_reason_code = "HEDGE_PROJECTION_ERROR"


class HedgeContractViolation(HedgeInvariantError):
    default_reason_code = "HEDGE_CONTRACT_VIOLATION"


class HedgeRetryableError(HedgeIntegrationError):
    default_reason_code = "HEDGE_RETRYABLE_ERROR"


class HedgeDefinitiveError(HedgeIntegrationError):
    default_reason_code = "HEDGE_DEFINITIVE_ERROR"
