"""
Class for generating a membrane with Toroidal shape
"""

import numpy as np
import scipy as sp
import math
from .mb import Mb
from .mb_error import MbError
from ...utils.affine import lin_map, tomo_rotate
from ...utils.tomo_utils import density_norm
from ...utils.poly import iso_surface, add_sfield_to_poly, poly_threshold


class MbEllipsoid(Mb):
    """
    Constructor

    :param tomo_shape: reference tomogram shape (X, Y and Z dimensions)
    :param v_size: reference tomogram voxel size (default 1)
    :param center: ellipsoid center (VERY IMPORTANT: coordinates are not in voxels)
    :param rot_q: rotation expressed as quaternion with respect ellipsoid center (default [1, 0, 0, 0] no rotation)
    :param thick: membrane thickness (default 1)
    :param layer_s: Gaussian sigma for each layer
    :param a: (default 1) semi axis length in X axis (before rotation)
    :param b: (default 1) semi axis length in Y axis (before rotation)
    :param c: (default 1) semi axis length in Z axis (before rotation)
    """

    def __init__(
        self,
        tomo_shape,
        v_size=1,
        center=(0, 0, 0),
        rot_q=(1, 0, 0, 0),
        thick=1,
        layer_s=1,
        a=1,
        b=1,
        c=1,
    ):
        """
        Constructor

        :param tomo_shape: reference tomogram shape (X, Y and Z dimensions)
        :param v_size: reference tomogram voxel size (default 1)
        :param center: ellipsoid center (VERY IMPORTANT: coordinates are not in voxels)
        :param rot_q: rotation expressed as quaternion with respect ellipsoid center (default [1, 0, 0, 0] no rotation)
        :param thick: membrane thickness (default 1)
        :param layer_s: Gaussian sigma for each layer
        :param a: (default 1) semi axis length in X axis (before rotation)
        :param b: (default 1) semi axis length in Y axis (before rotation)
        :param c: (default 1) semi axis length in Z axis (before rotation)
        """
        super(MbEllipsoid, self).__init__(
            tomo_shape, v_size, center, rot_q, thick, layer_s
        )
        assert (a > 0) and (b > 0) and (c > 0)
        self.__a, self.__b, self.__c = float(a), float(b), float(c)
        self._Mb__build_tomos()

    def _Mb__build_tomos(self):

        # Input parsing
        t_v, s_v = (
            0.5 * self._Mb__thick / self._Mb__v_size,
            self._Mb__layer_s / self._Mb__v_size,
        )
        a_v, b_v, c_v = (
            self.__a / self._Mb__v_size,
            self.__b / self._Mb__v_size,
            self.__c / self._Mb__v_size,
        )
        ao_v, bo_v, co_v = a_v + t_v, b_v + t_v, c_v + t_v
        ai_v, bi_v, ci_v = a_v - t_v, b_v - t_v, c_v - t_v
        ao_v_p1, bo_v_p1, co_v_p1 = ao_v + 1, bo_v + 1, co_v + 1
        ao_v_m1, bo_v_m1, co_v_m1 = ao_v - 1, bo_v - 1, co_v - 1
        ai_v_p1, bi_v_p1, ci_v_p1 = ai_v + 1, bi_v + 1, ci_v + 1
        ai_v_m1, bi_v_m1, ci_v_m1 = ai_v - 1, bi_v - 1, ci_v - 1
        p0_v = self._Mb__center / self._Mb__v_size

        # Generating the grid
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
            indexing="xy",
        )

        # Mask generation
        R_o = (
            ((X - p0_v[0]) / ao_v) ** 2
            + ((Y - p0_v[1]) / bo_v) ** 2
            + ((Z - p0_v[2]) / co_v) ** 2
        )
        R_i = (
            ((X - p0_v[0]) / ai_v) ** 2
            + ((Y - p0_v[1]) / bi_v) ** 2
            + ((Z - p0_v[2]) / ci_v) ** 2
        )
        self._Mb__mask = tomo_rotate(
            np.logical_and(R_i >= 1, R_o <= 1), self._Mb__rot_q, order=0
        )
        if self._Mb__mask.sum() == 0:
            raise MbError

        # Surface generation
        R_i = (
            ((X - p0_v[0]) / a_v) ** 2
            + ((Y - p0_v[1]) / b_v) ** 2
            + ((Z - p0_v[2]) / c_v) ** 2
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
        # lio.save_vtp(self._Mb__surf, './out/hold.vtp')
        self._Mb__surf = poly_threshold(
            self._Mb__surf, "mb_mask", mode="points", low_th=0.5
        )
        # lio.save_vtp(self._Mb__surf, './out/hold2.vtp')

        # Outer layer
        R_o = (
            ((X - p0_v[0]) / ao_v_p1) ** 2
            + ((Y - p0_v[1]) / bo_v_p1) ** 2
            + ((Z - p0_v[2]) / co_v_p1) ** 2
        )
        R_i = (
            ((X - p0_v[0]) / ao_v_m1) ** 2
            + ((Y - p0_v[1]) / bo_v_m1) ** 2
            + ((Z - p0_v[2]) / co_v_m1) ** 2
        )
        G = tomo_rotate(
            np.logical_and(R_i >= 1, R_o <= 1), self._Mb__rot_q, order=0
        )

        # Inner layer
        R_o = (
            ((X - p0_v[0]) / ai_v_p1) ** 2
            + ((Y - p0_v[1]) / bi_v_p1) ** 2
            + ((Z - p0_v[2]) / ci_v_p1) ** 2
        )
        R_i = (
            ((X - p0_v[0]) / ai_v_m1) ** 2
            + ((Y - p0_v[1]) / bi_v_m1) ** 2
            + ((Z - p0_v[2]) / ci_v_m1) ** 2
        )
        G += tomo_rotate(
            np.logical_and(R_i >= 1, R_o <= 1), self._Mb__rot_q, order=0
        )

        # Smoothing
        self._Mb__tomo = lin_map(
            density_norm(
                sp.ndimage.gaussian_filter(G.astype(float), s_v), inv=True
            ),
            ub=0,
            lb=1,
        )
