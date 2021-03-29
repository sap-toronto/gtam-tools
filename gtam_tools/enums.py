from collections import namedtuple
from enum import Enum, IntEnum

_PHF = namedtuple('PHF', ['AUTO', 'LIGHT', 'MEDIUM', 'HEAVY'])


class _BaseEnum(Enum):  # Extending functionality of the Enum class a bit
    @classmethod
    def as_dict(cls):
        return {k: v.value for k, v in cls.__members__.items()}

    def __eq__(self, other):  # For comparisons (e.g. PCEFactors.MEDIUM == 1.75 returns True)
        return self.value == other

    def __get__(self, obj, type=None):  # For retrieving values (eg.g var = PCEFactors.MEDIUM ==> var = 1.75)
        return self.value


class _BaseIntEnum(IntEnum):  # Extending functionality of the IntEnum class a bit
    @classmethod
    def as_dict(cls):
        return {k: v.value for k, v in cls.__members__.items()}


class CordonCountDir(_BaseIntEnum):
    NORTH = 1
    SOUTH = 3
    EAST = 2
    WEST = 4


class OccupationCategories(_BaseEnum):
    G = 'General Office/Clerical'
    M = 'Manufacturing/Construction/Trades'
    P = 'Professional/Management/Technical'
    S = 'Retail Sales and Service'


class PCEFactors(_BaseEnum):
    MEDIUM = 1.75
    HEAVY = 2.5


class PHFactors(Enum):
    # Notation of PHFactors: _PHF(Auto, Light, Medium, Heavy)
    AM = _PHF(0.469, 0.375068449, 0.419647872, 0.362607633)
    MD = _PHF(0.16666667, 0.177191204, 0.171328985, 0.176942152)
    PM = _PHF(0.307, 0.284041959, 0.325994799, 0.317223218)
    EV = _PHF(0.2, 0.2, 0.2, 0.2)


class StudentCategories(_BaseEnum):
    P = 'Primary'
    S = 'Secondary'
    U = 'Post-Secondary'


class TimeFormat(_BaseEnum):
    MINUTE_DELTA = 'minute_delta'
    COLON_SEP = 'colon_separated'


class ZoneNums(_BaseIntEnum):
    MAX_INTERNAL = 6000
    ROAMING = 8888
