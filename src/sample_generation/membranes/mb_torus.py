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


class MbTorus(Mb):

    def __init__(
        self,
        tomo_shape,
        v_size=1,
        center=(0, 0, 0),
        rot_q=(1, 0, 0, 0),
        thick=1,
        layer_s=1,
        rad_a=1,
        rad_b=1,
    ):
        """
        Constructor

        :param tomo_shape: reference tomogram shape (X, Y and Z dimensions)
        :param v_size: reference tomogram voxel size (default 1)
        :param center: ellipsoid center (VERY IMPORTANT: coordinates are not in voxels)
        :param rot_q: rotation expressed as quaternion with respect ellipsoid center (default [1, 0, 0, 0] no rotation)
        :param thick: membrane thickness (default 1)
        :param layer_s: Gaussian sigma for each layer
        :param rad_a: (default 1) torus radius
        :param rad_b: (default 1) torus tube radius
        """
        super(MbTorus, self).__init__(
            tomo_shape, v_size, center, rot_q, thick, layer_s
        )
        assert (rad_a > 0) and (rad_b > 0)
        self.__rad_a, self.__rad_b = float(rad_a), float(rad_b)
        self._Mb__build_tomos()

    def _Mb__build_tomos(self):

        # Input parsing
        t_v, s_v = (
            0.5 * (self._Mb__thick / self._Mb__v_size),
            self._Mb__layer_s / self._Mb__v_size,
        )
        rad_a_v, rad_b_v = (
            self.__rad_a / self._Mb__v_size,
            self.__rad_b / self._Mb__v_size,
        )
        bo_v, bi_v = rad_b_v + t_v, rad_b_v - t_v
        bo_v_p1, bo_v_m1 = bo_v + 1, bo_v - 1
        bi_v_p1, bi_v_m1 = bi_v + 1, bi_v - 1
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
            indexing="xy",
        )

        # Mask generation
        R_o = (
            (rad_a_v - np.sqrt((X - p0_v[0]) ** 2 + (Y - p0_v[1]) ** 2)) ** 2
            + (Z - p0_v[2]) ** 2
            - bo_v * bo_v
        ) <= 1
        R_i = (
            (rad_a_v - np.sqrt((X - p0_v[0]) ** 2 + (Y - p0_v[1]) ** 2)) ** 2
            + (Z - p0_v[2]) ** 2
            - bi_v * bi_v
        ) >= 1
        self._Mb__mask = tomo_rotate(
            np.logical_and(R_i, R_o), self._Mb__rot_q, order=0
        )

        # Surface generation
        R_i = (
            (rad_a_v - np.sqrt((X - p0_v[0]) ** 2 + (Y - p0_v[1]) ** 2)) ** 2
            + (Z - p0_v[2]) ** 2
            - rad_b_v * rad_b_v
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
            (rad_a_v - np.sqrt((X - p0_v[0]) ** 2 + (Y - p0_v[1]) ** 2)) ** 2
            + (Z - p0_v[2]) ** 2
            - bo_v_p1 * bo_v_p1
        ) <= 1
        R_i = (
            (rad_a_v - np.sqrt((X - p0_v[0]) ** 2 + (Y - p0_v[1]) ** 2)) ** 2
            + (Z - p0_v[2]) ** 2
            - bo_v_m1 * bo_v_m1
        ) >= 1
        G = tomo_rotate(np.logical_and(R_i, R_o), self._Mb__rot_q, order=0)

        # Inner layer
        R_o = (
            (rad_a_v - np.sqrt((X - p0_v[0]) ** 2 + (Y - p0_v[1]) ** 2)) ** 2
            + (Z - p0_v[2]) ** 2
            - bi_v_p1 * bi_v_p1
        ) <= 1
        R_i = (
            (rad_a_v - np.sqrt((X - p0_v[0]) ** 2 + (Y - p0_v[1]) ** 2)) ** 2
            + (Z - p0_v[2]) ** 2
            - bi_v_m1 * bi_v_m1
        ) >= 1
        G += tomo_rotate(np.logical_and(R_i, R_o), self._Mb__rot_q, order=0)

        # Smoothing
        self._Mb__tomo = lin_map(
            density_norm(
                sp.ndimage.gaussian_filter(G.astype(float), s_v), inv=True
            ),
            ub=0,
            lb=1,
        )
