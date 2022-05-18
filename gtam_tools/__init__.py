from ._version import get_versions
from .microsim_data import MicrosimData

__version__ = get_versions()['version']
del get_versions

from balsa.logging import init_root

init_root('gtam_tools')
del init_root
