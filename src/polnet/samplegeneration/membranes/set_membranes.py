import random

import numpy as np
import vtk

from polnet.samplegeneration.membranes.mb import Mb
from polnet.samplegeneration.membranes.mb_error import MbError
from polnet.samplegeneration.membranes.mb_sphere import MbSphere
from polnet.samplegeneration.membranes.membrane_generator import MemGen
from polnet.utils.distribution import gen_rand_unit_quaternion
from polnet.utils.poly import poly_scale

MAX_TRIES_MB = 10


class SetMembranes:
    """Class for modelling a set of membranes (same kind) within a tomogram."""

    def __init__(
        self,
        voi: np.ndarray,
        v_size: float,
        gen_rnd_surfs: MemGen,
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

        # Input parsing
        assert isinstance(voi, np.ndarray) and (voi.dtype == bool)
        # TODO: assert issubclass(gen_rnd_surfs.__class__, SurfGen)
        # assert (
        #     hasattr(param_rg, "__len__")
        #     and (len(param_rg) == 3)
        #     and (param_rg[0] <= param_rg[1])
        # )
        # assert (
        #     hasattr(thick_rg, "__len__")
        #     and (len(thick_rg) == 2)
        #     and (thick_rg[0] <= thick_rg[1])
        # )
        # assert (
        #     hasattr(layer_rg, "__len__")
        #     and (len(layer_rg) == 2)
        #     and (layer_rg[0] <= layer_rg[1])
        # )
        # assert (occ >= 0) and (occ <= 100)
        # assert (over_tolerance >= 0) and (over_tolerance <= 100)
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
        # self.__param_rg, self.__thick_rg, self.__layer_rg = (
        #     param_rg,
        #     thick_rg,
        #     layer_rg,
        # )
        # self.__occ, self.__over_tolerance = occ, over_tolerance
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
            raise MbError("Input membrane volume is zero")
        if (over_tolerance is None) or (
            not self.check_overlap(mb, over_tolerance)
        ):
            # Density tomogram insertion
            mb.insert_density_svol(self.__tomo, merge=merge, mode="tomo")
            # Ground Truth
            mb.insert_density_svol(self.__gtruth, merge="max", mode="mask")
            # VTK PolyData surface insertion
            self.__surfs = poly_scale(
                vtk.vtkAppendPolyData()
                .AddInputData(self.__surfs)
                .AddInputData(mb.surf)
                .GetOutput(),
                scale_factor=1.0,
            )
            self.__count_mbs += 1

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
            hold_mb = self.__gen_rnd_surfs.gen_surface()
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
                        f"Membrane {count_mb} inserted, total occupancy: {self.mb_occupancy}, volume: {hold_mb.vol}. Details:\n\t{hold_mb}"
                    )
                count_mb += 1
            except MbError:
                count_exp += 1
                if count_exp == MAX_TRIES_MB:
                    print(
                        f"WARNING: more than {MAX_TRIES_MB} tries failed to insert a membrane!"
                    )
                    break
            count_exp = 0


#     Class for modelling a set of membranes within a tomogram
#     """

#     def __init__(
#         self,
#         voi,
#         v_size,
#         gen_rnd_surfs,
#         param_rg,
#         thick_rg,
#         layer_rg,
#         occ,
#         over_tolerance=0,
#         bg_voi=None,
#         grow=0,
#     ):
#         """
#         Construction

#         :param voi: a 3D numpy array to define a VOI (Volume Of Interest) for membranes
#         :param v_size: voxel size
#         :param gen_rnd_surf: an of object that inherits from lrandom.SurfGen class to generate random instances with
#                              membrane surface parameters, therefore the objects class determine the shape of the membranes
#                              generated
#         :param param_rg: 3-tuple with parameters relative to membrane geometry (min_radius, max_radius,
#                          max_eccentricity)
#         :param thick_rg: membrane thickness range (2-tuple)
#         :param layer_s: lipid layer range (2-tuple)
#         :param occ: occupancy threshold in percentage [0, 100]%
#         :param over_tolerance: fraction of overlapping tolerance for self avoiding (default 0, in range [0,1))
#         :param bg_voi: background VOI (Default None), if present membrane areas in this VOI will be removed and
#         :param grow: number of voxel to grow the VOI
#         not considered for overlapping. It must be binary with the same shape of voi
#         """

#         # Input parsing
#         assert isinstance(voi, np.ndarray) and (voi.dtype == bool)
#         # TODO: assert issubclass(gen_rnd_surfs.__class__, SurfGen)
#         assert (
#             hasattr(param_rg, "__len__")
#             and (len(param_rg) == 3)
#             and (param_rg[0] <= param_rg[1])
#         )
#         assert (
#             hasattr(thick_rg, "__len__")
#             and (len(thick_rg) == 2)
#             and (thick_rg[0] <= thick_rg[1])
#         )
#         assert (
#             hasattr(layer_rg, "__len__")
#             and (len(layer_rg) == 2)
#             and (layer_rg[0] <= layer_rg[1])
#         )
#         assert (occ >= 0) and (occ <= 100)
#         assert (over_tolerance >= 0) and (over_tolerance <= 100)
#         if bg_voi is not None:
#             assert isinstance(bg_voi, np.ndarray) and (bg_voi.dtype == bool)
#             assert (
#                 len(voi.shape) == len(bg_voi.shape)
#                 and voi.shape == bg_voi.shape
#             )
#         assert grow >= 0

#         # Variables assignment
#         self.__voi = voi
#         self.__bg_voi = None
#         if bg_voi is not None:
#             self.__bg_voi = bg_voi
#         self.__vol = (
#             float(self.__voi.sum()) * v_size * v_size * v_size
#         )  # without the float cast it may raise overflow warining in Windows
#         self.__v_size = v_size
#         self.__tomo, self.__gtruth = np.zeros(
#             shape=voi.shape, dtype=np.float16
#         ), np.zeros(shape=voi.shape, dtype=bool)
#         self.__surfs, self.__app_vtp = (
#             vtk.vtkPolyData(),
#             vtk.vtkAppendPolyData(),
#         )
#         self.__count_mbs = 0
#         self.__gen_rnd_surfs = gen_rnd_surfs
#         self.__param_rg, self.__thick_rg, self.__layer_rg = (
#             param_rg,
#             thick_rg,
#             layer_rg,
#         )
#         self.__occ, self.__over_tolerance = occ, over_tolerance
#         self.__grow = grow

#     def get_vol(self):
#         return self.__vol

#     def get_mb_occupancy(self):
#         return self.__gtruth.sum() / np.prod(
#             np.asarray(self.__voi.shape, dtype=float)
#         )

#     def build_set(self, verbosity=False):
#         """
#         Build a set of ellipsoid membranes and insert them in a tomogram and a vtkPolyData object

#         :param verbosity: if True (default False) the output message with info about the membranes generated is printed
#         :return:
#         """

#         # Initialization
#         count_mb = 1
#         count_exp = 0

#         # Network loop
#         while self.get_mb_occupancy() < self.__occ:

#             # Polymer initialization
#             p0 = np.asarray(
#                 (
#                     self.__voi.shape[0] * self.__v_size * random.random(),
#                     self.__voi.shape[1] * self.__v_size * random.random(),
#                     self.__voi.shape[2] * self.__v_size * random.random(),
#                 )
#             )
#             thick, layer_s = random.uniform(
#                 self.__thick_rg[0], self.__thick_rg[1]
#             ), random.uniform(self.__layer_rg[0], self.__layer_rg[1])
#             rot_q = gen_rand_unit_quaternion()

#             try:

#                 # Membrane generation according the predefined surface model
#                 # if isinstance(self.__gen_rnd_surfs, EllipGen):
#                 #     ellip_axes = self.__gen_rnd_surfs.gen_parameters_exp()
#                 #     hold_mb = MbEllipsoid(self.__voi.shape, v_size=self.__v_size,
#                 #                           center=p0, rot_q=rot_q, thick=thick, layer_s=layer_s,
#                 #                           a=ellip_axes[0], b=ellip_axes[1], c=ellip_axes[2])
#                 #     hold_rad = np.mean(ellip_axes)
#                 # el
#                 # if isinstance(self.__gen_rnd_surfs, SphGen):
#                 # rad = self.__gen_rnd_surfs.gen_parameters()
#                 rad = random.uniform(self.__param_rg[0], self.__param_rg[1])
#                 hold_mb = MbSphere(
#                     self.__voi.shape,
#                     v_size=self.__v_size,
#                     center=p0,
#                     thick=thick,
#                     layer_s=layer_s,
#                     rad=rad,
#                 )
#                 hold_rad = rad
#                 # elif isinstance(self.__gen_rnd_surfs, TorGen):
#                 #     tor_axes = self.__gen_rnd_surfs.gen_parameters()
#                 #     hold_mb = MbTorus(self.__voi.shape, v_size=self.__v_size,
#                 #                       center=p0, rot_q=rot_q, thick=thick, layer_s=layer_s,
#                 #                       rad_a=tor_axes[0], rad_b=tor_axes[1])
#                 #     hold_rad = np.mean(tor_axes)
#                 # else:
#                 #     print('ERROR: not valid random surface parameters generator: ' + str(self.__gen_rnd_surfs.__class__))
#                 #     raise MbError

#                 # Background masking
#                 if self.__bg_voi is not None:
#                     hold_mb.masking(self.__bg_voi)

#                 # Insert membrane
#                 self.insert_mb(
#                     hold_mb,
#                     merge="max",
#                     over_tolerance=self.__over_tolerance,
#                     check_vol=True,
#                     grow=self.__grow,
#                 )
#                 if verbosity:
#                     print(
#                         "Membrane "
#                         + str(count_mb)
#                         + ", total occupancy: "
#                         + str(self.get_mb_occupancy())
#                         + ", volume: "
#                         + str(hold_mb.vol)
#                         + ", thickness: "
#                         + str(hold_mb.thick)
#                         + ", layer_s: "
#                         + str(hold_mb.layer_s)
#                         + ", radius (avg): "
#                         + str(hold_rad)
#                     )
#                 count_mb += 1

#             # Handling the exception raised when a membrane could not be generated properly
#             except MbError:
#                 count_exp += 1
#                 # print('JOl')
#                 # print('Count: ' + str(count_exp))
#                 if count_exp == MAX_TRIES_MB:
#                     print(
#                         "WARNING: more than "
#                         + str(MAX_TRIES_MB)
#                         + " tries failed to insert a membrane!"
#                     )
#                     break
#             count_exp = 0

#     def get_voi(self):
#         """
#         Get the VOI

#         :return: an ndarray
#         """
#         return self.__voi

#     def get_tomo(self):
#         """
#         Get the tomogram with the membranes within the VOI

#         :return: an ndarray
#         """
#         # return np.invert(self.__voi) * self.__tomo
#         return self.__tomo

#     def get_gtruth(self):
#         """
#         Get the ground truth within the VOI

#         :return: an ndarray
#         """
#         # return self.__voi * self.__gtruth
#         return self.__gtruth

#     def get_vtp(self):
#         """
#         Get the set of membranes as a vtkPolyData with their surfaces

#         :return: a vtkPolyData
#         """
#         return self.__surfs

#     def get_num_mbs(self):
#         """
#         Get the number of membranes in the set

#         :return: an integer with the number of membranes
#         """
#         return self.__count_mbs

#     def check_overlap(self, mb, over_tolerance):
#         """
#         Determines if the membrane overlaps with any member within the membranes set

#         :param mb: input Membrane to check for the overlapping
#         :param over_tolerance: overlapping tolerance (percentage of membrane voxel overlapping)
#         """
#         mb_mask = mb.mask
#         # tomo_mb = np.zeros(shape=mb_mask.shape, dtype=bool)
#         # mb.insert_density_svol(tomo_mb, merge='max')
#         # tomo_over = np.logical_and(mb_mask, self.__gtruth)
#         tomo_over = np.logical_and(mb_mask, np.logical_not(self.__voi))
#         if 100.0 * (tomo_over.sum() / self.get_vol()) > over_tolerance:
#             return True
#         return False

#     def compute_overlap(self, mb):
#         """
#         Computes membrane overlapping with the set

#         :param mb: input Membrane to check for the overlapping
#         """
#         mb_mask = mb.mask
#         tomo_mb = np.zeros(shape=mb_mask.shape, dtype=bool)
#         mb.insert_density_svol(tomo_mb, merge="max")
#         tomo_over = np.logical_and(
#             np.logical_and(tomo_mb, self.__gtruth), self.__voi
#         )
#         return 100.0 * (tomo_over.sum() / self.get_vol())

#     def insert_mb(
#         self, mb, merge="min", over_tolerance=None, check_vol=True, grow=0
#     ):
#         """
#         Insert the membrane into the set (tomogram, vtkPolyData and Ground Truth)

#         :param mb: input membrane (Mb) object
#         :param merge: merging mode for density insertion, valid: 'min' (default), 'max', 'sum' and 'insert'
#         :param over_tolerance: overlapping tolerance (percentage of membrane voxel overlapping), if None then disabled
#         :param check_vol: if True (default) the input membrane volume is check and if equal to zero then the membrane
#                           is not inserted and MbError is raised
#         :param grow: number of voxel to grow the membrane tomogram to insert (default 0), only used in 'voi' mode
#         :return: raises a ValueError if the membrane is not inserted
#         """
#         if check_vol and (mb.vol <= 0):
#             raise MbError("Input membrane volume is zero")
#         if (over_tolerance is None) or (
#             not self.check_overlap(mb, over_tolerance)
#         ):
#             # Density tomogram insertion
#             mb.insert_density_svol(self.__tomo, merge=merge, mode="tomo")
#             # Ground Truth
#             mb.insert_density_svol(self.__gtruth, merge="max", mode="mask")
#             # VOI
#             mb.insert_density_svol(
#                 self.__voi, merge="min", mode="voi", grow=grow
#             )
#             # Surfaces insertion
#             self.__app_vtp.AddInputData(mb.vtp)
#             self.__app_vtp.Update()
#             self.__surfs = poly_scale(self.__app_vtp.GetOutput(), self.__v_size)
#             self.__count_mbs += 1
#         else:
#             raise MbError
