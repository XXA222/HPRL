"""One-way configuration migration into the Clean Mainline schema.

Only raw configuration input may contain the retired ``hedge.r56`` key.  The
runtime never consumes it.  Normalization moves the mapping to
``hedge.operations`` before JSON-schema validation and then removes the legacy
key from the in-memory configuration.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from freqtrade.exceptions import OperationalException


LEGACY_OPERATIONS_KEY = "r56"
CURRENT_OPERATIONS_KEY = "operations"



def has_legacy_operations_alias(config: object) -> bool:
    """Return whether a raw config still contains the retired operations key."""

    if not isinstance(config, MutableMapping):
        return False
    hedge = config.get("hedge")
    return isinstance(hedge, MutableMapping) and LEGACY_OPERATIONS_KEY in hedge

def migrate_legacy_operations_alias(
    hedge: MutableMapping[str, Any],
) -> bool:
    """Move the retired operations key to the canonical key exactly once.

    Returns ``True`` only when a migration was performed.  Ambiguous input
    fails closed; the function never merges two independently supplied
    mappings.
    """

    if LEGACY_OPERATIONS_KEY not in hedge:
        return False

    if CURRENT_OPERATIONS_KEY in hedge:
        raise OperationalException(
            "hedge.operations and the retired operations key cannot both be configured."
        )

    legacy = hedge[LEGACY_OPERATIONS_KEY]
    if not isinstance(legacy, MutableMapping):
        raise OperationalException(
            "The retired Hedge operations configuration must be a JSON object."
        )

    hedge[CURRENT_OPERATIONS_KEY] = legacy
    del hedge[LEGACY_OPERATIONS_KEY]
    return True
