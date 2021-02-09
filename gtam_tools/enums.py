from enum import Enum, IntEnum


class SpecialZones(IntEnum):
    ROAMING = 8888


class TimeFormat(Enum):

    MINUTE_DELTA = 'minute_delta'
    COLON_SEP = 'colon_separated'
