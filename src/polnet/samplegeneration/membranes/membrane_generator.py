import random

from abc import ABC, abstractmethod


class MemGen(ABC):
    """
    Abstract class for generating membrane surfaces with random parameters
    """

    def __init__(
        self,
        thick_rg: tuple[float, float],
        layer_s_rg: tuple[float, float],
        occ_rg: tuple[float, float],
        over_tol: float,
        mb_den_cf_rg: tuple[float, float],
    ) -> None:
        """
        Constructor

        Args:
            thick_rg (tuple[float, float]): tuple with the min and max thickness values.
            layer_s_rg (tuple[float, float]): tuple with the min and max layer sigma values.
            occ_rg (tuple[float, float]): tuple with the min and max occupancy values.
            over_tol (float): overlap tolerance for the membrane set.
            mb_den_cf_rg (tuple[float, float]): tuple with the min and max membrane density contrast values.

        Returns:
            None
        """

        self.__thick_rg = thick_rg
        self.__layer_s_rg = layer_s_rg
        self.__occ_rg = occ_rg
        self.__over_tol = over_tol
        self.__mb_den_cf_rg = mb_den_cf_rg

    @property
    def occ(self) -> float:
        """
        Returns a random occupancy value within the defined range
        """
        return random.uniform(self.__occ_rg[0], self.__occ_rg[1])

    @property
    def over_tolerance(self) -> float:
        """
        Returns the overlap tolerance value
        """
        return self.__over_tol

    @abstractmethod
    def generate(self):
        """
        Generates a membrane with random parameters
        """
        raise NotImplemented
