"""Classes for generating microtubular structures."""

from .hn import Hn, HnGen
from .hn_factory import HnFactory

class HnMt(Hn):
    pass

@HnFactory.register("mt")
class MtGen(HnGen):
    pass