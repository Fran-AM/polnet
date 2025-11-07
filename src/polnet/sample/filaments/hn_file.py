"""Module for handling helicoidal network configuration files."""

import ast
from pathlib import Path

class HnFile:
    """
    For handling helicoidal network configuration files
    """ 

    def __init__(self):
        self.__params = {}

    @property
    def type(self):
        return self.__params.get("HN_TYPE", None)

    def load(self, in_file: Path) -> None:
        """
        Load helicoidal network parameters from an input file

        Args:
            in_file (Path): path to the input file with extension .hns
        """
        if not in_file.suffix == ".hns":
            raise ValueError(f"Input file {in_file} does not have .hns extension.")
        if not in_file.exists():
            raise FileNotFoundError(f"Helical network file {in_file} does not exist.") 
        with open(in_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "#" in line:
                    line = line.split("#")[
                        0
                    ].strip()
                if "=" in line:
                    key, value = [part.strip() for part in line.split("=", 1)]
                    try:
                        self.__params[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        self.__params[key] = (
                            value
                        )
        return self.__params.copy()