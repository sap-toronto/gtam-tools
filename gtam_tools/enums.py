from enum import Enum, IntEnum


class PCEFactors(Enum):
    MEDIUM = 1.7
    HEAVY = 2.5


class TimeFormat(Enum):
    MINUTE_DELTA = 'minute_delta'
    COLON_SEP = 'colon_separated'


class ZoneNums(IntEnum):
    MAX_INTERNAL = 6000
    ROAMING = 8888
