import random

import numpy as np

from polnet.samplegeneration.membranes.membrane_generator import MemGen
from polnet.samplegeneration.membranes.membrane_factory import MembraneFactory
from polnet.samplegeneration.membranes.mb_sphere import MbSphere


@MembraneFactory.register("sphere")
class SphGen(MemGen):
    """
    Class for model the parameters for modelling an Sphere
    """

    def __init__(
        self,
        thick_rg: tuple[float, float],
        layer_s_rg: tuple[float, float],
        occ_rg: tuple[float, float],
        over_tol: float,
        mb_den_cf_rg: tuple[float, float],
        min_rad: float,
        max_rad: float = None,
    ) -> None:
        """
        Constructor

        Args:
            radius_rg: tuple with the min and max radius values

        """

        super(SphGen, self).__init__(
            thick_rg=thick_rg,
            layer_s_rg=layer_s_rg,
            occ_rg=occ_rg,
            over_tol=over_tol,
            mb_den_cf_rg=mb_den_cf_rg,
        )
        self._min_rad = min_rad
        self._max_rad = max_rad

    @classmethod
    def from_params(cls, params: dict) -> "SphGen":
        """
        Creates a SphGen object from a dictionary of parameters

        Args:
            params: dictionary with the membrane parameters

        """

        thick_rg = params.get("MB_THICK_RG", (25.0, 35.0))
        layer_s_rg = params.get("MB_LAYER_S_RG", (0.5, 2.0))
        occ_rg = params.get("MB_OCC_RG", (0.001, 0.003))
        over_tol = params.get("MB_OVER_TOL", 0.0)
        mb_den_cf_rg = params.get("MB_DEN_CF_RG", (0.3, 0.5))

        min_rad = params.get("MB_MIN_RAD", 75.0)
        max_rad = params.get("MB_MAX_RAD", None)

        return cls(
            thick_rg=thick_rg,
            layer_s_rg=layer_s_rg,
            occ_rg=occ_rg,
            over_tol=over_tol,
            mb_den_cf_rg=mb_den_cf_rg,
            min_rad=min_rad,
            max_rad=max_rad,
        )

    def generate(self, voi_shape: tuple, v_size: float):
        """
        Generates a spherical membrane with random parameters within the input volume of interest shape

        Args:
            voi_shape: shape of the volume of interest
            v_size: voxel size
        """
        radius = random.uniform(self.__radius_rg[0], self.__radius_rg[1])
        center = np.asarray(
            (
                voi_shape[0] * v_size * random.random(),
                voi_shape[1] * v_size * random.random(),
                voi_shape[2] * v_size * random.random(),
            )
        )
        thick = random.uniform(
            self._SurfGen__thick_rg[0], self._SurfGen__thick_rg[1]
        )
        layer_s = random.uniform(
            self._SurfGen__layer_s_rg[0], self._SurfGen__layer_s_rg[1]
        )

        return MbSphere(
            voi_shape=voi_shape,
            v_size=v_size,
            thick=thick,
            layer_s=layer_s,
            center=center,
            radius=radius,
        )
