import math
from abc import ABC, abstractmethod

import numpy as np

from ...utils.utils import (
    lin_map
)

from ...utils.poly import (
    poly_scale,
    poly_translate,
    iso_surface
)

class FiberUnit(ABC):
    """
    Abstract class to generate fiber unit (set of monomers)
    """

    @abstractmethod
    def get_vtp(self):
        raise NotImplementedError

    @abstractmethod
    def get_tomo(self):
        raise NotImplementedError


class FiberUnitSDimer(FiberUnit):
    """
    Class for modeling a fiber unit as dimer of two spheres
    """

    def __init__(self, sph_rad, v_size=1):
        """
        Constructor

        :param sph_rad: radius for spheres
        :param v_size: voxel size (default 1)
        """
        assert (sph_rad > 0) and (v_size > 0)
        self.__sph_rad, self.__v_size = float(sph_rad), float(v_size)
        self.__size = int(math.ceil(6.0 * (sph_rad / v_size)))
        if self.__size % 2 != 0:
            self.__size += 1
        self.__tomo, self.__surf = None, None
        self.__gen_sdimer()

    def get_vtp(self):
        return self.__surf

    def get_tomo(self):
        return self.__tomo

    def __gen_sdimer(self):
        """
        Contains the procedure to generate the Dimer of spheres with the specified size by using logistic functions
        """

        # Input parsing
        sph_rad_v = self.__sph_rad / self.__v_size
        sph_rad_v2 = sph_rad_v * sph_rad_v * 0.5625  # (0.75*rad)^2

        # Generating the grid
        self.__tomo = np.zeros(
            shape=(self.__size, self.__size, self.__size), dtype=np.float32
        )
        dx, dy, dz = (
            float(self.__tomo.shape[0]),
            float(self.__tomo.shape[1]),
            float(self.__tomo.shape[2]),
        )
        dx2, dy2, dz2 = (
            math.floor(0.5 * dx),
            math.floor(0.5 * dy),
            math.floor(0.5 * dz),
        )
        x_l, y_l, z_l = -dx2, -dy2, -dz2
        x_h, y_h, z_h = -dx2 + dx, -dy2 + dy, -dz2 + dz
        X, Y, Z = np.meshgrid(
            np.arange(x_l, x_h),
            np.arange(y_l, y_h),
            np.arange(z_l, z_h),
            indexing="xy",
        )
        X += 0.5
        Y += 0.5
        Z += 0.5
        # X, Y, Z = X.astype(np.float16), Y.astype(np.float16), X.astype(np.float16)

        # from polnet import lio

        # Generate the first unit
        Yh = Y + sph_rad_v
        R = X * X + Yh * Yh + Z * Z - sph_rad_v2

        # lio.write_mrc(R.astype(np.float32), '/fs/pool/pool-lucic2/antonio/polnet/riboprot/synth_all/hold_R1.mrc')

        self.__tomo += 1.0 / (1.0 + np.exp(-R))

        # lio.write_mrc(self.__tomo.astype(np.float32), '/fs/pool/pool-lucic2/antonio/polnet/riboprot/synth_all/hold_R2.mrc')

        # Generate the second unit
        Yh = Y - sph_rad_v
        R = X * X + Yh * Yh + Z * Z - sph_rad_v2
        self.__tomo += 1.0 / (1.0 + np.exp(-R))

        # Generating the surfaces
        self.__tomo = lin_map(
            self.__tomo, lb=1, ub=0
        )  # self.__tomo = lin_map(self.__tomo, lb=0, ub=1)
        self.__surf = iso_surface(
            self.__tomo, 0.25
        )  # self.__surf = iso_surface(self.__tomo, .75)

        # lio.write_mrc(self.__tomo, '/fs/pool/pool-lucic2/antonio/polnet/riboprot/synth_all/hold_funit1.mrc')
        # lio.save_vtp(self.__surf, '/fs/pool/pool-lucic2/antonio/polnet/riboprot/synth_all/hold_funit1.vtp')

        self.__surf = poly_scale(self.__surf, self.__v_size)
        self.__surf = poly_translate(
            self.__surf,
            -0.5 * self.__v_size * (np.asarray(self.__tomo.shape) - 0.5),
        )


class MTUnit(FiberUnit):
    """
    Class for modelling a fiber unit for microtubules (MTs)
    """

    def __init__(self, sph_rad=40, mt_rad=100.5, n_units=13, v_size=1):
        """
        Constructor

        :param sph_rad: radius for spheres (default 40, approximate tubulin radius in A)
        :param mt_rad: microtubule radius (default 100.5, approximate microtubule radius in A)
        :param n_units: number of units (default 13, number of protofilaments that compund a MT)
        :param v_size: voxel size (default 1)
        """
        assert (sph_rad > 0) and (mt_rad > 0) and (n_units > 0) and (v_size > 0)
        self.__sph_rad, self.__mt_rad, self.__n_units, self.__v_size = (
            float(sph_rad),
            float(mt_rad),
            int(n_units),
            float(v_size),
        )
        self.__size = int(math.ceil(6.0 * (sph_rad / v_size))) - 6
        if self.__size % 2 != 0:
            self.__size += 1
        self.__tomo, self.__surf = None, None
        self.__gen_sdimer()

    def get_vtp(self):
        return self.__surf

    def get_tomo(self):
        return self.__tomo

    def __gen_sdimer(self):
        """
        Contains the procedure to generate the Dimer of spheres with the specified size by using logistic functions
        """

        # Input parsing
        sph_rad_v, mt_rad_v = (
            self.__sph_rad / self.__v_size,
            self.__mt_rad / self.__v_size,
        )
        sph_rad_v2 = sph_rad_v * sph_rad_v * 0.25  # (0.9*rad)^2

        # Generating the grid
        self.__tomo = np.zeros(
            shape=(self.__size, self.__size, self.__size), dtype=np.float32
        )
        dx, dy, dz = (
            float(self.__tomo.shape[0]),
            float(self.__tomo.shape[1]),
            float(self.__tomo.shape[2]),
        )
        dx2, dy2, dz2 = (
            math.floor(0.5 * dx),
            math.floor(0.5 * dy),
            math.floor(0.5 * dz),
        )
        x_l, y_l, z_l = -dx2, -dy2, -dz2
        x_h, y_h, z_h = -dx2 + dx, -dy2 + dy, -dz2 + dz
        X, Y, Z = np.meshgrid(
            np.arange(x_l, x_h),
            np.arange(y_l, y_h),
            np.arange(z_l, z_h),
            indexing="xy", # Indexing xy inside the local grid
        )
        X += 0.5
        Y += 0.5
        Z += 0.5
        # X, Y, Z = X.astype(np.float16), Y.astype(np.float16), X.astype(np.float16)

        # Loop for generate the units
        Z2 = Z * Z
        ang_step = 2.0 * np.pi / self.__n_units
        ang = ang_step
        while ang <= 2.0 * np.pi:
            # Generate the unit
            # x, y = mt_rad_v * math.cos(ang), mt_rad_v * math.sin(ang)
            x, y = mt_rad_v * math.cos(ang), mt_rad_v * math.sin(ang)
            Xh, Yh = X + x, Y + y
            # R = np.abs(Xh * Xh + Yh * Yh + Z2)
            R = Xh * Xh + Yh * Yh + Z2 - sph_rad_v2
            # from polnet import lio
            # lio.write_mrc(R.astype(np.float32), '/fs/pool/pool-lucic2/antonio/polnet/riboprot/synth_all/hold_R1.mrc')
            F = 1.0 / (1.0 + np.exp(-R))
            # mask_F = F < 0.1
            self.__tomo += -F + 1
            ang += ang_step

        # from polnet import lio
        # lio.write_mrc(self.__tomo, '/fs/pool/pool-lucic2/antonio/polnet/riboprot/synth_all/hold_mtunit1.mrc')

        # Generating the surfaces
        self.__tomo = lin_map(
            self.__tomo, lb=0, ub=1
        )  # self.__tomo = lin_map(self.__tomo, lb=0, ub=1)
        self.__surf = iso_surface(
            self.__tomo, 0.25
        )  # self.__surf = iso_surface(self.__tomo, .75)
        self.__surf = poly_scale(self.__surf, self.__v_size)
        self.__surf = poly_translate(
            self.__surf,
            -0.5 * self.__v_size * (np.asarray(self.__tomo.shape) - 0.5),
        )

        # lio.save_vtp(self.__surf, '/fs/pool/pool-lucic2/antonio/polnet/riboprot/synth_all/hold_mtunit1.vtp')
