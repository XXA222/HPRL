"""Hedge ML/RL regression package."""

import numpy as np


# Mirror the project-wide tests/conftest.py floating-point policy even when the
# isolated ML/RL gate uses --confcutdir and intentionally skips the root conftest.
np.seterr(all="raise")
