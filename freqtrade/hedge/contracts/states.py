from enum import StrEnum


class ReadinessState(StrEnum):
    BOOTSTRAPPING = "BOOTSTRAPPING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    HALT = "HALT"
