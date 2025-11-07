"""Classes for generating actin structures."""

from .hn import Hn, HnGen
from .hn_factory import HnFactory

class HnActin(Hn):
    pass

@HnFactory.register("actin")
class ActGen(HnGen):
    pass