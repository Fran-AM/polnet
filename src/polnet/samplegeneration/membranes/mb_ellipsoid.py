"""Class for generating a membrane with ellipsoidal shape"""

import numpy as np
import scipy as sp
import math
from .mb import Mb
from .mb_error import MbError
from polnet.utils.affine import lin_map, tomo_rotate
from polnet.utils.tomo_utils import density_norm
from polnet.utils.poly import iso_surface, add_sfield_to_poly, poly_threshold


class MbEllipsoid(Mb):
    """Class for generating a membrane with Ellipsoidal shape"""

    def __init__(
        self,
        tomo_shape: tuple,
        v_size: float = 1,
        thick: float = 1,
        layer_s: float = 1,
        center: tuple[float, float, float] = (0, 0, 0),
        rot_q: tuple[float, float, float, float] = (1, 0, 0, 0),
        a: float = 1,
        b: float = 1,
        c: float = 1,
    ) -> None:
        """
        Constructor

        Args:
            tomo_shape (tuple): reference tomogram shape (X, Y and Z dimensions)
            v_size (float, optional): reference tomogram voxel size in angstroms. Defaults to 1.
            thick (float, optional): membrane thickness in angstroms. Defaults to 1.
            layer_s (float, optional): Gaussian sigma for each layer in angstroms. Defaults to 1.
            center (tuple, optional): center of the ellipsoid in angstroms. Defaults to (0, 0, 0).
            rot_q (tuple, optional): rotation quaternion defining the ellipsoid orientation. Defaults to (1, 0, 0, 0).
            a (float, optional): semi-axis a of the ellipsoid in angstroms. Defaults to 1.
            b (float, optional): semi-axis b of the ellipsoid in angstroms. Defaults to 1.
            c (float, optional): semi-axis c of the ellipsoid in angstroms. Defaults to 1.

        Raises:
            TypeError: if 'tomo_shape' is not a tuple of three integers
            ValueError: if any dimension of 'tomo_shape' is not an integer
            ValueError: if 'v_size' or 'thick' are not positive floats or 'layer_s' is negative
            TypeError: if 'center' is not a tuple of three floats
            ValueError: if any dimension of 'center' is not a float
            TypeError: if 'rot_q' is not a tuple of four floats
            ValueError: if any dimension of 'rot_q' is not a float
            TypeError: if 'a', 'b' or 'c' are not floats
            ValueError: if 'a', 'b' or 'c' are not positive

        Returns:
            None
        """
        super(MbEllipsoid, self).__init__(tomo_shape, v_size, thick, layer_s)

        if not hasattr(center, "__len__") or (len(center) != 3):
            raise TypeError(
                "center must be a tuple of three floats (X, Y and Z)"
            )
        if not all(isinstance(c, (int, float)) for c in center):
            raise TypeError("All dimensions of center must be floats")
        if not hasattr(rot_q, "__len__") or (len(rot_q) != 4):
            raise TypeError(
                "rot_q must be a tuple of four floats (w, x, y and z components)"
            )
        if not all(isinstance(r, (int, float)) for r in rot_q):
            raise TypeError("All dimensions of rot_q must be floats")
        if not isinstance(a, (int, float)):
            raise TypeError("a must be a float")
        if not isinstance(b, (int, float)):
            raise TypeError("b must be a float")
        if not isinstance(c, (int, float)):
            raise TypeError("c must be a float")
        if a <= 0 or b <= 0 or c <= 0:
            raise ValueError("a, b and c must be positive")

        self.__a, self.__b, self.__c = float(a), float(b), float(c)
        self._Mb__build_tomos()

    def _Mb__build_tomos(self):
        """Generates the tomogram, mask and surface of the ellipsoidal membrane

        Args:
            None

        Returns:
            None
        """

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
