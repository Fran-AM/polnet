"""Module to define abstract membrane class and a custom membrane error exception"""

from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
import vtk

from polnet.utils.poly import poly_mask
from polnet.utils.tomo_utils import insert_svol_tomo


class Mb(ABC):
    """Abstract class to model membranes with different geometries

    A membrane is modelled as two parallel surfaces with Gaussian profile
    """

    def __init__(
        self,
        voi_shape: tuple[int, int, int],
        v_size: float = 1,
        thick: float = 1,
        layer_s: float = 1,
    ) -> None:
        """Constructor

        Defines the basic properties of a membrane, but does not generate it.

        Args:
            voi_shape (tuple): reference volume of interest shape (X, Y and Z dimensions)
            v_size (float, optional): reference volume of interest voxel size in angstroms. Defaults to 1.
            thick (float, optional): membrane thickness in angstroms. Defaults to 1.
            layer_s (float, optional): Gaussian sigma for each layer in angstroms. Defaults to 1.

        Raises:
            TypeError: if 'voi_shape' is not a tuple of three integers
            ValueError: if any dimension of 'voi_shape' is not an integer
            ValueError: if 'v_size' or 'thick' are not positive floats or 'layer_s' is negative

        Returns:
            None
        """
        if not hasattr(voi_shape, "__len__") or (len(voi_shape) != 3):
            raise TypeError(
                "voi_shape must be a tuple of three integers (X, Y and Z dimensions)"
            )
        if not all(isinstance(dim, int) for dim in voi_shape):
            raise TypeError("All dimensions of voi_shape must be integers")
        if v_size <= 0:
            raise ValueError("v_size must be a positive float")
        if thick <= 0:
            raise ValueError("thick must be a positive float")
        if layer_s < 0:
            raise ValueError("layer_s must be a non negative float")

        self.__voi_shape = voi_shape
        self.__v_size = float(v_size)
        self.__thick, self.__layer_s = float(thick), float(layer_s)
        self.__density, self.__mask, self.__surf = None, None, None

    @property
    def v_size(self) -> float:
        """Get voxel size

        Returns:
            float: voxel size in angstroms
        """
        return self.__v_size

    @property
    def thick(self) -> float:
        """Get membrane thickness, bilayer gap

        Returns:
            float: membrane thickness in angstroms
        """
        return self.__thick

    @property
    def layer_s(self) -> float:
        """Get Gaussian sigma for each layer

        Returns:
            float: layer sigma in angstroms
        """
        return self.__layer_s

    @property
    def vol(self) -> float:
        """Get the polymer volume

        Returns:
            float: volume in cubic angstroms
        """
        return self.__mask.sum() * self.__v_size**3

    @property
    def density(self) -> np.ndarray:
        """Get the membrane density

        Returns:
            np.ndarray: a numpy 3D array representing the membrane density
        """
        return self.__density

    @property
    def mask(self) -> np.ndarray:
        """Get the membrane binary mask

        Returns:
            np.ndarray: a binary numpy 3D array representing the membrane mask
        """
        return self.__mask

    @property
    def vtp(self) -> vtk.vtkPolyData:
        """Get the membrane as an VTK surface

        Returns:
            vtk.vtkPolyData: the membrane surface
        """
        return self.__surf

    def masking(self, mask: np.ndarray) -> None:
        """Removes membrane voxels in an external mask. Membrane voxels at mask 0-valued positions will be set to 0.

        Args:
            mask (np.ndarray): the input external mask. A binary ndarray with the same shape as the membrane volume of interest.

        Raises:
            TypeError: if 'mask' is not a binary numpy ndarray
            ValueError: if 'mask' does not have the same shape as the membrane volume of interest

        Returns:
            None
        """
        if not isinstance(mask, np.ndarray) or mask.dtype != bool:
            raise TypeError("mask must be a binary numpy ndarray")
        if (len(mask.shape) != len(self.__tomo.shape)) or (
            mask.shape != self.__tomo.shape
        ):
            raise ValueError(
                "mask must have the same shape as the membrane volume of interest"
            )  
        self.__density[mask] = 0
        self.__mask[mask] = False
        self.__surf = poly_mask(self.__surf, mask)

    def insert_density_svol(
        self, tomo: np.ndarray, merge="max", mode="tomo", grow=0
    ) -> None:
        """Insert a membrane into a tomogram

        Args:
            tomo: tomogram where the membrane is added
            merge: merging mode, valid: 'min' (default), 'max', 'sum' and 'insert'
            mode: determines which data are inserted, valid: 'tomo' (default), 'mask' and 'voi'
            grow: number of voxel to grow the membrane tomogram to insert (default 0), only used in 'voi' mode

        Raises:
            TypeError: if 'tomo' is not a 3D numpy ndarray
            ValueError: if 'tomo' does not have the same shape as the membrane tomogram
            ValueError: if 'merge' is not 'min', 'max', 'sum' or 'insert'
            ValueError: if 'mode' is not 'tomo', 'mask' or 'voi'

        Returns:
            None
        """
        if not isinstance(tomo, np.ndarray) or (len(tomo.shape) != 3):
            raise TypeError("tomo must be a 3D numpy ndarray")

        if (len(tomo.shape) != len(self.__voi_shape)) or (
            tomo.shape != self.__voi_shape
        ):
            raise ValueError(
                "tomo must have the same shape as the membrane tomogram"
            )

        if not (merge in ["min", "max", "sum", "insert"]):
            raise ValueError("merge must be 'min', 'max', 'sum' or 'insert'")
        if not (mode in ["tomo", "mask", "voi"]):
            raise ValueError("mode must be 'tomo', 'mask' or 'voi'")

        if mode == "tomo":
            hold = self.__density
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
            hold, tomo, 0.5 * np.asarray(self.__voi_shape), merge=merge
        )

    @abstractmethod
    def __build(self) -> None:
        """Generates the membrane within a tomogram.

        Must be called at the end of the constructor of each subclass.

        Raises:
            NotImplementedError: if the subclass does not implement this method
        """
        raise NotImplementedError("Mb subclasses must implement this method")


class MbError(Exception):
    """Custom exception for membrane-related errors.

    Attributes:
        message (str): Description of the error.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
