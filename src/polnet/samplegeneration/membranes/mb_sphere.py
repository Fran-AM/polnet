"""Class for generating a membrane with Spherical shape"""

import math

import numpy as np
import scipy as sp

from polnet.samplegeneration.membranes.mb import Mb
from polnet.utils.affine import lin_map, tomo_rotate
from polnet.utils.tomo_utils import density_norm
from polnet.utils.poly import iso_surface, add_sfield_to_poly, poly_threshold


class MbSphere(Mb):
    """Class for generating a membrane with Spherical shape"""

    def __init__(
        self,
        tomo_shape: tuple,
        v_size: float = 1,
        thick: float = 1,
        layer_s: float = 1,
        center: tuple[float, float, float] = (0, 0, 0),
        rad: float = 1,
    ) -> None:
        """Constructor

        Defines the basic properties of a spherical membrane, and generates it.

        Args:
            tomo_shape (tuple): reference tomogram shape (X, Y and Z dimensions)
            v_size (float, optional): reference tomogram voxel size in angstroms. Defaults to 1.
            thick (float, optional): membrane thickness in angstroms. Defaults to 1.
            layer_s (float, optional): Gaussian sigma for each layer in angstroms. Defaults to 1.
            center (tuple, optional): center of the sphere in angstroms. Defaults to (0, 0, 0).
            rad (float, optional): radius of the sphere in angstroms. Defaults to 1.

        Raises:
            TypeError: if 'tomo_shape' is not a tuple of three integers
            ValueError: if any dimension of 'tomo_shape' is not an integer
            ValueError: if 'v_size' or 'thick' are not positive floats or 'layer_s' is negative
            TypeError: if 'center' is not a tuple of three floats
            ValueError: if any dimension of 'center' is not a float
            TypeError: if 'rad' is not a float
            ValueError: if 'rad' is not positive

        Returns:
            None
        """
        super(MbSphere, self).__init__(tomo_shape, v_size, thick, layer_s)

        if not hasattr(center, "__len__") or (len(center) != 3):
            raise TypeError(
                "center must be a tuple of three floats (X, Y and Z)"
            )
        if not all(isinstance(c, (int, float)) for c in center):
            raise TypeError("All dimensions of center must be floats")
        if not isinstance(rad, (int, float)):
            raise TypeError("rad must be a float")
        if rad <= 0:
            raise ValueError("rad must be positive")

        self.__center = np.array([float(c) for c in center])
        self.__rad = float(rad)
        self.__rot_q = np.array([1, 0, 0, 0])  # No rotation TODO: remove
        self._Mb__build_tomos()

    def _Mb__build_tomos(self):
        """Generates the tomogram, mask and surface of the spherical membrane

        Args:
            None

        Returns:
            None
        """

        # Input parsing
        t_v, s_v = (
            0.5 * (self._Mb__thick / self._Mb__v_size),
            self._Mb__layer_s / self._Mb__v_size,
        )
        rad_v = self.__rad / self._Mb__v_size
        ao_v = rad_v + t_v
        ai_v = rad_v - t_v
        ao_v_p1 = ao_v + 1
        ao_v_m1 = ao_v - 1
        ai_v_p1 = ai_v + 1
        ai_v_m1 = ai_v - 1
        p0_v = self.__center / self._Mb__v_size

        # Generating the bilayer
        dx, dy, dz = (
            float(self._Mb__tomo_shape[0]),
            float(self._Mb__tomo_shape[1]),
            float(self._Mb__tomo_shape[2]),
        )
        dx2, dy2, dz2 = (
            math.floor(0.5 * dx),
            math.floor(0.5 * dy),
            math.floor(0.5 * dz),
        )
        p0_v[0] -= dx2
        p0_v[1] -= dy2
        p0_v[2] -= dz2
        x_l, y_l, z_l = -dx2, -dy2, -dz2
        x_h, y_h, z_h = -dx2 + dx, -dy2 + dy, -dz2 + dz
        X, Y, Z = np.meshgrid(
            np.arange(x_l, x_h),
            np.arange(y_l, y_h),
            np.arange(z_l, z_h),
            indexing="ij",
        )

        # Mask generation
        R_o = (
            ((X - p0_v[0]) / ao_v) ** 2
            + ((Y - p0_v[1]) / ao_v) ** 2
            + ((Z - p0_v[2]) / ao_v) ** 2
        )
        R_i = (
            ((X - p0_v[0]) / ai_v) ** 2
            + ((Y - p0_v[1]) / ai_v) ** 2
            + ((Z - p0_v[2]) / ai_v) ** 2
        )

        self._Mb__mask = np.logical_and(R_i >= 1, R_o <= 1)
        # self._Mb__mask = tomo_rotate(
        #     np.logical_and(R_i >= 1, R_o <= 1), self.__rot_q, order=0
        # )

        # Surface generation
        R_i = (
            ((X - p0_v[0]) / rad_v) ** 2
            + ((Y - p0_v[1]) / rad_v) ** 2
            + ((Z - p0_v[2]) / rad_v) ** 2
        )
        # R_i = tomo_rotate(R_i, self.__rot_q, mode="reflect")
        self._Mb__surf = iso_surface(R_i, 1)
        add_sfield_to_poly(
            self._Mb__surf,
            self._Mb__mask,
            "mb_mask",
            dtype="int",
            interp="NN",
            mode="points",
        )
        self._Mb__surf = poly_threshold(
            self._Mb__surf, "mb_mask", mode="points", low_th=0.5
        )

        # Outer layer
        R_o = (
            ((X - p0_v[0]) / ao_v_p1) ** 2
            + ((Y - p0_v[1]) / ao_v_p1) ** 2
            + ((Z - p0_v[2]) / ao_v_p1) ** 2
        )
        R_i = (
            ((X - p0_v[0]) / ao_v_m1) ** 2
            + ((Y - p0_v[1]) / ao_v_m1) ** 2
            + ((Z - p0_v[2]) / ao_v_m1) ** 2
        )
        G = np.logical_and(R_i >= 1, R_o <= 1)
        # G = tomo_rotate(
        #     np.logical_and(R_i >= 1, R_o <= 1), self.__rot_q, order=0
        # )
        # R = (X - p0_v[0])**2 + (Y - p0_v[1])**2 + (Z - p0_v[2])**2
        # G = tomo_rotate(np.logical_and(R >= ao_v_m1**2, R <= ao_v_p1**2), self.__rot_q, order=0)

        # Inner layer
        R_o = (
            ((X - p0_v[0]) / ai_v_p1) ** 2
            + ((Y - p0_v[1]) / ai_v_p1) ** 2
            + ((Z - p0_v[2]) / ai_v_p1) ** 2
        )
        R_i = (
            ((X - p0_v[0]) / ai_v_m1) ** 2
            + ((Y - p0_v[1]) / ai_v_m1) ** 2
            + ((Z - p0_v[2]) / ai_v_m1) ** 2
        )
        G += np.logical_and(R_i >= 1, R_o <= 1)
        # G += tomo_rotate(
        #     np.logical_and(R_i >= 1, R_o <= 1), self.__rot_q, order=0
        # )
        # G += tomo_rotate(np.logical_and(R >= ai_v_m1**2, R_o <= ai_v_p1**2), self._Mb__rot_q, order=0)

        # Smoothing
        # TODO: is it required the density_norm() having lin_map()?
        # self._Mb__tomo = lin_map(density_norm(sp.ndimage.gaussian_filter(G.astype(float), s_v), inv=True), ub=0, lb=1)
        self._Mb__tomo = lin_map(
            -1 * sp.ndimage.gaussian_filter(G.astype(float), s_v), ub=0, lb=1
        )
