"""HPRL-specific exception hierarchy."""

from __future__ import annotations


class HPRLError(RuntimeError):
    """Base exception for the isolated HPRL subsystem."""


class HPRLConfigError(HPRLError, ValueError):
    """Raised when an HPRL configuration is internally inconsistent."""


class HPRLDependencyError(HPRLError, ImportError):
    """Raised when an optional HPRL dependency is unavailable."""


class HPRLShapeError(HPRLError, ValueError):
    """Raised when a tensor or array violates a public HPRL shape contract."""


class HPRLRiskError(HPRLError, ValueError):
    """Raised when an unsafe action cannot be projected into the configured risk envelope."""
