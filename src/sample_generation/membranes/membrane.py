import numpy as np
import scipy as sp
from abc import ABC, abstractmethod

from src.utils.poly import poly_mask
from src.utils.arrays import insert_svol_tomo


class Mb(ABC):
    """
    Abstract class to model membranes with different geometries
    """

    def __init__(
        self,
        tomo_shape,
        v_size=1,
        center=(0, 0, 0),
        rot_q=(1, 0, 0, 0),
        thick=1,
        layer_s=1,
    ):
        """
        Constructor

        :param tomo_shape: reference tomogram shape (X, Y and Z dimensions)
        :param v_size: reference tomogram voxel size (default 1)
        :param center: ellipsoid center (VERY IMPORTANT: coordinates are not in voxels)
        :param rot_q: rotation expressed as quaternion with respect ellipsoid center (default [1, 0, 0, 0] no rotation)
        :param thick: membrane thickness (default 1)
        :param layer_s: Gaussian sigma for each layer
        """
        assert hasattr(tomo_shape, "__len__") and (len(tomo_shape) == 3)
        assert v_size > 0
        assert (thick > 0) and (layer_s > 0)
        assert hasattr(center, "__len__") and (len(center) == 3)
        assert hasattr(rot_q, "__len__") and (len(rot_q) == 4)
        self.__tomo_shape, self.__v_size = tomo_shape, v_size
        self.__center, self.__rot_q = np.asarray(
            center, dtype=float
        ), np.asarray(rot_q, dtype=float)
        self.__thick, self.__layer_s = float(thick), float(layer_s)
        self.__tomo, self.__mask, self.__surf = None, None, None

    @property
    def thick(self):
        """
        Get membrane thickness, bilayer gap

        :return: thickness as a float
        """
        return self.__thick

    @property
    def layer_s(self):
        """
        Get Gaussian sigma for each layer

        :return: layer sigma as a float
        """
        return self.__layer_s

    @property
    def vol(self):
        """
        Get the polymer volume

        :param fast: if True (default) the volume monomer is only computed once
        :return: the computed volume
        """
        return self.__mask.sum() * self.__v_size**3

    @property
    def tomo(self):
        """
        Get the membrane within a tomogram

        :return: a numpy 3D array
        """
        return self.__tomo

    @property
    def mask(self):
        """
        Get the membrane within a binary tomogram

        :return: a binary numpy 3D array
        """
        return self.__mask

    @property
    def vtp(self):
        """
        Get the membrane as an VTK surface

        :return: a vtkPolyData object
        """
        return self.__surf

    def masking(self, mask):
        """
        Removes membrane voxels in an external mask

        :param mask: the input external mask, binary ndarray with the same shape as the membrane tomogram, tomogram
        voxels at mask 0-valued positions will be set to 0
        :return: None
        """
        assert isinstance(mask, np.ndarray) and mask.dtype == bool
        assert (
            len(mask.shape) == len(self.__tomo.shape)
            and mask.shape == self.__tomo.shape
        )
        mask_ids = np.invert(mask)
        self.__tomo[mask_ids] = 0
        self.__mask[mask_ids] = False
        self.__surf = poly_mask(self.__surf, mask)

    def insert_density_svol(self, tomo, merge="max", mode="tomo", grow=0):
        """
        Insert a membrane into a tomogram

        :param tomo: tomogram where m_svol is added
        :param merge: merging mode, valid: 'min' (default), 'max', 'sum' and 'insert'
        :param mode: determines which data are inserted, valid: 'tomo' (default), 'mask' and 'voi'
        :param grow: number of voxel to grow the membrane tomogram to insert (default 0), only used in 'voi' mode
        :return:
        """
        assert (mode == "tomo") or (mode == "mask") or (mode == "voi")
        if mode == "tomo":
            hold = self.__tomo
        elif mode == "mask":
            hold = self.__mask
        elif mode == "voi":
            if grow >= 1:
                hold = np.invert(
                    sp.ndimage.morphology.binary_dilation(
                        self.__mask, iterations=grow
                    )
                )
            else:
                hold = np.invert(self.__mask)
        insert_svol_tomo(
            hold, tomo, 0.5 * np.asarray(self.__tomo.shape), merge=merge
        )

    @abstractmethod
    def __build_tomos(self):
        """
        Generates the membrane within a tomogram

        :return: the generated tomogram and its binary mask
        """
        raise NotImplemented
