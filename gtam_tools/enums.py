from enum import Enum, IntEnum


class ZoneNums(IntEnum):
    MAX_INTERNAL = 6000
    ROAMING = 8888


class TimeFormat(Enum):

    MINUTE_DELTA = 'minute_delta'
    COLON_SEP = 'colon_separated'
