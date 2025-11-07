from abc import ABC, abstractmethod

class Hn(ABC):
    pass

class HnGen(ABC):
    pass

class HnError(Exception):
    """Custom exception for helicoidal network-related errors.

    Attributes:
        message (str): Description of the error.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)