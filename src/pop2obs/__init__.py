"""Synthetic observations from aerosol populations."""

from .instruments import list_instruments, observe
from .records import Observation

__version__ = "0.1.0.dev0"

__all__ = ["Observation", "__version__", "list_instruments", "observe"]
