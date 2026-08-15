"""Strategy-side validation helpers for the clean-mainline Hedge contract."""
from __future__ import annotations
from typing import Iterable

from freqtrade.hedge.strategies.contract import HEDGE_SIGNAL_COLUMNS

class HedgeStrategyMixin:
    """Validate Hedge columns without coupling a strategy to the execution engine."""
    hedge_allowed_columns=frozenset(HEDGE_SIGNAL_COLUMNS)
    @classmethod
    def validate_hedge_dataframe(cls,dataframe:object)->None:
        columns=set(getattr(dataframe,"columns",()))
        required={"open","high","low","close","volume"}
        missing=required-columns
        if missing:raise ValueError("Hedge strategy dataframe is missing: "+", ".join(sorted(missing)))
        unknown={name for name in columns if str(name).startswith("hedge_") and name not in cls.hedge_allowed_columns}
        if unknown:raise ValueError("unknown Hedge signal column(s): "+", ".join(sorted(unknown)))
        preferred={"hedge_long_score","hedge_short_score"}
        legacy={"enter_long","enter_short"}
        if not preferred.issubset(columns) and not legacy.issubset(columns):
            raise ValueError("strategy must emit Hedge scores or legacy enter_long/enter_short")
    @classmethod
    def hedge_contract_columns(cls)->tuple[str,...]:return tuple(HEDGE_SIGNAL_COLUMNS)
