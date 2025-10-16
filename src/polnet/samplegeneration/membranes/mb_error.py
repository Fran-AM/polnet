class MbError(Exception):
    """Custom exception for membrane-related errors.

    Attributes:
        message (str): Description of the error.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        """String representation of the error."""
        return f"MbError: {self.message}"

    def __repr__(self) -> str:
        """Official string representation of the error."""
        return f"MbError({self.message})"
