import sys

import numpy as np
import vtk

from polnet.samplegeneration.membranes.mb import Mb, MbError
from polnet.samplegeneration.membranes.mb_generator import MbGen
from polnet.utils.poly import poly_scale


class SetMembranes:
    """Class for modelling a set of membranes (same kind) within a tomogram."""

    #: Maximum number of tries to insert a membrane
    MAX_TRIES_MB: int = 10

    def __init__(
        self,
        voi: np.ndarray,
        v_size: float,
        gen_rnd_surfs: MbGen,
        bg_voi: np.ndarray = None,
        grow: int = 0,
    ) -> None:
        """
        Constructor

        Setting up the parameters for generating a set of membranes within a tomogram.

        Args:
            voi (np.ndarray): A 3D numpy array to define a VOI (Volume Of Interest) for membranes.
            v_size (float): Voxel size (in angstroms).
            gen_rnd_surfs: An object that inherits from SurfGen class to generate random instances with
                             membrane surface parameters, therefore the objects class determine the shape of the membranes
                             generated.
            bg_voi (np.ndarray, optional): Background VOI (Default None), if present membrane areas in this VOI will be removed and
                                            not considered for overlapping. It must be binary with the same shape of voi.
            grow (int, optional): Number of voxel to grow the VOI.
        """

        if bg_voi is not None:
            assert isinstance(bg_voi, np.ndarray) and (bg_voi.dtype == bool)
            assert (
                len(voi.shape) == len(bg_voi.shape)
                and voi.shape == bg_voi.shape
            )
        assert grow >= 0

        # Variables assignment
        self.__voi = voi
        self.__bg_voi = None
        if bg_voi is not None:
            self.__bg_voi = bg_voi
        self.__vol = (
            float(self.__voi.sum()) * v_size * v_size * v_size
        )  # without the float cast it may raise overflow warining in Windows
        self.__v_size = v_size
        self.__tomo, self.__gtruth = np.zeros(
            shape=voi.shape, dtype=np.float16
        ), np.zeros(shape=voi.shape, dtype=bool)
        self.__surfs, self.__app_vtp = (
            vtk.vtkPolyData(),
            vtk.vtkAppendPolyData(),
        )
        self.__count_mbs = 0
        self.__gen_rnd_surfs = gen_rnd_surfs
        self.__grow = grow

    @property
    def vol(self) -> float:
        """Get the volume of the VOI."""
        return self.__vol

    @property
    def mb_occupancy(self) -> float:
        """Get the membrane occupancy within the VOI."""
        return self.__gtruth.sum() / np.prod(
            np.asarray(self.__voi.shape, dtype=float)
        )

    @property
    def voi(self) -> np.ndarray:
        """Get the VOI.

        Returns:
            np.ndarray: The volume of interest.
        """
        return self.__voi

    @property
    def tomo(self) -> np.ndarray:
        """Get the tomogram with the membranes within the VOI.

        Returns:
            np.ndarray: The tomogram.
        """
        return self.__tomo

    @property
    def gtruth(self) -> np.ndarray:
        """Get the ground truth within the VOI.

        Returns:
            np.ndarray: The ground truth.
        """
        return self.__gtruth

    @property
    def vtp(self) -> vtk.vtkPolyData:
        """Get the set of membranes as a vtkPolyData with their surfaces.

        Returns:
            vtk.vtkPolyData: The set of membranes.
        """
        return self.__surfs

    @property
    def num_mbs(self) -> int:
        """Get the number of membranes in the set.

        Returns:
            int: The number of membranes.
        """
        return self.__count_mbs

    def check_overlap(self, mb: Mb, over_tolerance: float) -> bool:
        """
        Determines if the membrane overlaps with any member within the membranes set.

        Args:
            mb (Mb): Input Membrane to check for the overlapping.
            over_tolerance (float): Overlapping tolerance (percentage of membrane voxel overlapping).

        Returns:
            bool: True if there is an overlap, False otherwise.
        """

        mb_mask = mb.mask
        tomo_over = np.logical_and(mb_mask, np.logical_not(self.__voi))
        if 100.0 * (tomo_over.sum() / mb.vol) > over_tolerance:
            return True
        return False

    def compute_overlap(self, mb: Mb) -> float:
        """
        Computes membrane overlapping with the set.

        Args:
            mb (Mb): Input Membrane to check for the overlapping.

        Returns:
            float: The percentage of overlap.
        """

        mb_mask = mb.mask
        tomo_mb = np.zeros(shape=mb_mask.shape, dtype=bool)
        mb.insert_density_svol(tomo_mb, merge="max")
        tomo_over = np.logical_and(
            np.logical_and(tomo_mb, self.__gtruth), self.__voi
        )
        return 100.0 * (tomo_over.sum() / self.vol)

    def insert_mb(
        self,
        mb: Mb,
        merge: str = "min",
        over_tolerance: float = None,
        check_vol: bool = True,
        grow: int = 0,
    ) -> None:
        """
        Insert the membrane into the set (tomogram, vtkPolyData and Ground Truth).

        Args:
            mb (Mb): Input membrane (Mb) object.
            merge (str, optional): Merging mode for density insertion, valid: 'min' (default), 'max', 'sum' and 'insert'.
            over_tolerance (float, optional): Overlapping tolerance (percentage of membrane voxel overlapping), if None then disabled.
            check_vol (bool, optional): If True (default), check for volume consistency before insertion.
            grow (int, optional): Number of voxels to grow the membrane before insertion.

        Raises:
            MbError: MbError if the membrane is not inserted.

        Returns:
            None
        """
        if check_vol and (mb.vol <= 0):
            raise MbError("Membrane volume is zero or negative, cannot be inserted.")
        if (over_tolerance is None) or (not self.check_overlap(mb, over_tolerance)):
            # Density tomogram insertion
            mb.insert_density_svol(self.__tomo, merge=merge, mode='tomo')
            # Ground Truth
            mb.insert_density_svol(self.__gtruth, merge='max', mode='mask')
            # VOI
            mb.insert_density_svol(self.__voi, merge='min', mode='voi', grow=grow)
            # Surfaces insertion
            self.__app_vtp.AddInputData(mb.vtp)
            self.__app_vtp.Update()
            self.__surfs = poly_scale(self.__app_vtp.GetOutput(), self.__v_size)
            self.__count_mbs += 1
        else:
            raise MbError("Membrane overlaps with the set, cannot be inserted.")

    def build_set(self, verbosity: bool = False) -> None:
        """
        Build a set of membranes and insert them in a tomogram and a vtkPolyData object.

        Args:
            verbosity (bool, optional): If True (default False) the output message with info about the membranes generated is printed.

        Returns:
            None
        """

        # Initialization
        count_mb = 1
        count_exp = 0
        max_occ = self.__gen_rnd_surfs.occ
        over_tolerance = self.__gen_rnd_surfs.over_tolerance

        while self.mb_occupancy < max_occ:
            hold_mb = self.__gen_rnd_surfs.generate(voi_shape=self.__voi.shape, v_size=self.__v_size)
            if self.__bg_voi is not None:
                hold_mb.masking(self.__bg_voi)
            try:
                self.insert_mb(
                    hold_mb,
                    merge="max",
                    over_tolerance=over_tolerance,
                    check_vol=True,
                    grow=self.__grow,
                )
                if verbosity:
                    print(
                        f"Membrane {count_mb} inserted, total occupancy: {self.mb_occupancy:.3f}, volume: {hold_mb.vol} Å. Details:\n\t{hold_mb}"
                    )
                count_mb += 1
            except MbError:
                count_exp += 1
                if count_exp == self.MAX_TRIES_MB:
                    print(
                        f"WARNING: more than {self.MAX_TRIES_MB} tries failed to insert a membrane!",
                        file=sys.stderr,
                    )
                    break
            count_exp = 0

