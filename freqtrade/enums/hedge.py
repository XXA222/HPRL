
from enum import Enum


class _ValueStrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class PositionMode(_ValueStrEnum):
    ONEWAY = "oneway"
    HEDGE = "hedge"


class PositionSide(_ValueStrEnum):
    BOTH = "BOTH"
    LONG = "LONG"
    SHORT = "SHORT"


class PositionAction(_ValueStrEnum):
    OPEN = "OPEN"
    INCREASE = "INCREASE"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"

    @property
    def increases_risk(self) -> bool:
        return self in {self.OPEN, self.INCREASE}

    @property
    def reduces_risk(self) -> bool:
        return self in {self.REDUCE, self.CLOSE}
