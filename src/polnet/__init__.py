"""Polnet: synthetic cryo-ET tomogram generation toolkit.

Top-level package exposing the primary public API for generating
synthetic cryo-electron tomograms, including membranes, filaments,
and macro-molecular complexes.

:author: Antonio Martinez-Sanchez
:maintainer: Juan Diego Gallego Nicolás
"""

import logging
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("polnet")
except PackageNotFoundError:
    __version__ = "unknown"

from .logging_conf import (
    _LOGGER as logger,
    setup_logger,
)
from .pipeline import gen_tomos
from .sample import (
    MbFile,
    SyntheticSample,
)
from .tomogram import SynthTomo

logging.getLogger("polnet").addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    "SyntheticSample",
    "MbFile",
    "SynthTomo",
    "gen_tomos",
    "setup_logger",
    "logger",
]
