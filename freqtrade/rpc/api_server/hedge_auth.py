"""FastAPI dependency wrappers for core Hedge control authentication."""

from __future__ import annotations

from typing import Callable

from fastapi import Depends, HTTPException

from freqtrade.hedge.control.auth import (
    ConfirmationService,
    HedgePrincipal,
    HedgeRole,
)


def require_role(
    minimum: HedgeRole,
    principal_dependency: Callable[..., HedgePrincipal],
):
    try:
        role = minimum if isinstance(minimum, HedgeRole) else HedgeRole(minimum)
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum role is invalid") from exc
    if not callable(principal_dependency):
        raise TypeError("principal_dependency must be callable")

    def dependency(
        principal: HedgePrincipal = Depends(principal_dependency),
    ) -> HedgePrincipal:
        if not isinstance(principal, HedgePrincipal):
            raise HTTPException(status_code=401, detail="invalid hedge principal")
        if principal.role < role:
            raise HTTPException(status_code=403, detail="insufficient hedge role")
        return principal

    return dependency


__all__ = [
    "ConfirmationService",
    "HedgePrincipal",
    "HedgeRole",
    "require_role",
]
