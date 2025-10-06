"""
Class for generating a membrane with Spherical shape
"""

import numpy as np
import scipy as sp
import math
from .mb import Mb
from .mb_error import MbError
from ...utils.affine import lin_map, tomo_rotate
from ...utils.tomo_utils import density_norm
from ...utils.poly import iso_surface, add_sfield_to_poly, poly_threshold


class MbSphere(Mb):

    def __init__(
        self,
        tomo_shape,
        v_size=1,
        center=(0, 0, 0),
        rot_q=(1, 0, 0, 0),
        thick=1,
        layer_s=1,
        rad=1,
    ):
        """
        Constructor

        :param tomo_shape: reference tomogram shape (X, Y and Z dimensions)
        :param v_size: reference tomogram voxel size (default 1)
        :param center: ellipsoid center (VERY IMPORTANT: coordinates are not in voxels)
        :param rot_q: rotation expressed as quaternion with respect ellipsoid center (default [1, 0, 0, 0] no rotation)
        :param thick: membrane thickness (default 1)
        :param layer_s: Gaussian sigma for each layer
        :param rad: (default 1) sphere radius
        """
        super(MbSphere, self).__init__(
            tomo_shape, v_size, center, rot_q, thick, layer_s
        )
        assert rad > 0
        self.__rad = float(rad)
        self._Mb__build_tomos()

    def _Mb__build_tomos(self):

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
        p0_v = self._Mb__center / self._Mb__v_size

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
        self._Mb__mask = tomo_rotate(
            np.logical_and(R_i >= 1, R_o <= 1), self._Mb__rot_q, order=0
        )

        # Surface generation
        R_i = (
            ((X - p0_v[0]) / rad_v) ** 2
            + ((Y - p0_v[1]) / rad_v) ** 2
            + ((Z - p0_v[2]) / rad_v) ** 2
        )
        R_i = tomo_rotate(R_i, self._Mb__rot_q, mode="reflect")
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
        G = tomo_rotate(
            np.logical_and(R_i >= 1, R_o <= 1), self._Mb__rot_q, order=0
        )
        # R = (X - p0_v[0])**2 + (Y - p0_v[1])**2 + (Z - p0_v[2])**2
        # G = tomo_rotate(np.logical_and(R >= ao_v_m1**2, R <= ao_v_p1**2), self._Mb__rot_q, order=0)

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
        G += tomo_rotate(
            np.logical_and(R_i >= 1, R_o <= 1), self._Mb__rot_q, order=0
        )
        # G += tomo_rotate(np.logical_and(R >= ai_v_m1**2, R_o <= ai_v_p1**2), self._Mb__rot_q, order=0)

        # Smoothing
        # TODO: is it required the density_norm() having lin_map()?
        # self._Mb__tomo = lin_map(density_norm(sp.ndimage.gaussian_filter(G.astype(float), s_v), inv=True), ub=0, lb=1)
        self._Mb__tomo = lin_map(
            -1 * sp.ndimage.gaussian_filter(G.astype(float), s_v), ub=0, lb=1
        )
