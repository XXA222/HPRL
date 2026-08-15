"""Retired SQL-adapter location for the execution domain.

Persistence-backed execution adapters live in
the persistence adapter package.  The execution package stays
database-implementation agnostic and therefore intentionally exports no SQL class
from this module.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
