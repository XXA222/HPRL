from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_OVERLAY = _PACKAGE_ROOT / "overlay"
if _OVERLAY.is_dir():
    sys.path.insert(0, str(_OVERLAY))
