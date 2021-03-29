from collections import namedtuple
from enum import Enum, IntEnum

_PHF = namedtuple('PHF', ['AUTO', 'LIGHT', 'MEDIUM', 'HEAVY'])


class CordonCountDir(IntEnum):
    NORTH = 1
    SOUTH = 3
    EAST = 2
    WEST = 4


class PCEFactors(Enum):
    MEDIUM = 1.75
    HEAVY = 2.5


class PHFactors(Enum):
    # Notation of PHFactors: _PHF(Auto, Light, Medium, Heavy)
    AM = _PHF(0.469, 0.375068449, 0.419647872, 0.362607633)
    MD = _PHF(0.16666667, 0.177191204, 0.171328985, 0.176942152)
    PM = _PHF(0.307, 0.284041959, 0.325994799, 0.317223218)
    EV = _PHF(0.2, 0.2, 0.2, 0.2)


class TimeFormat(Enum):
    MINUTE_DELTA = 'minute_delta'
    COLON_SEP = 'colon_separated'


class ZoneNums(IntEnum):
    MAX_INTERNAL = 6000
    ROAMING = 8888
