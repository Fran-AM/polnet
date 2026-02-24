import math
import random

from ..polymers import Network
import numpy as np
import vtk
from .flms_fiber import HelixFiber
from .flms_gen import FlmsParamGen, HxParamGenBranched

from ...utils.utils import (
    points_distance
)

from ...utils.poly import point_to_poly

from ...utils.affine import poly_translate
from dataclasses import dataclass, field

NET_TYPE_STR = "net_type"

class NetHelixFiber(Network):
    """
    Class for generating a network of isolated helix fibers, unconnected and randomly distributed
    """

    def __init__(
        self,
        voi,
        v_size,
        l_length,
        m_surf,
        gen_hfib_params,
        occ,
        min_p_len,
        hp_len,
        mz_len,
        mz_len_f,
        over_tolerance=0,
        unit_diam=None,
    ):
        """
        Construction

        :param voi: a 3D numpy array to define a VOI (Volume Of Interest) for polymers
        :param v_size: voxel size (default 1)
        :param l_length: polymer link length
        :param m_surf: monomer surf
        :param gen_hfib_params: a instance of a random generation model (random.PGen) to obtain random fiber
        parametrization
        :param min_p_len: minimum persistence length
        :param hp_len: helix period length
        :param mz_len: monomer length
        :param mz_len_f: maximum length factor in z-axis
        :param occ: occupancy threshold in percentage [0, 100]%
        :param over_tolerance: fraction of overlapping tolerance for self avoiding (default 0, in range [0,1))
        :param unit_diam: structural unit diameter
        """

        # Initialize abstract variables
        super(NetHelixFiber, self).__init__(voi, v_size)

        # Input parsing
        assert l_length > 0
        assert isinstance(m_surf, vtk.vtkPolyData)
        assert isinstance(gen_hfib_params, FlmsParamGen)
        assert (occ >= 0) and (occ <= 100)
        assert (over_tolerance >= 0) and (over_tolerance <= 100)
        assert (
            (min_p_len > 0)
            and (hp_len > 0)
            and (mz_len > 0)
            and (mz_len_f >= 0)
        )

        # Variables assignment
        self.__l_length, self.__m_surf = l_length, m_surf
        self.__gen_hfib_params = gen_hfib_params
        self.__occ, self.__over_tolerance = occ, over_tolerance
        self.__min_p_len, self.__hp_len = min_p_len, hp_len
        self.__mz_len, self.__mz_len_f = mz_len, mz_len_f
        self.__unit_diam = unit_diam

    def build_network(self):
        """
        Add helix fibres until an occupancy limit is passed

        :return:
        """

        MAX_TRIES = 1000
        tries_count = 0

        # Network loop
        while self._Network__pl_occ < self.__occ and tries_count < MAX_TRIES:
            tries_count += 1

            # Polymer initialization
            p0 = np.asarray(
                (
                    self._Network__voi.shape[0]
                    * self._Network__v_size
                    * random.random(),
                    self._Network__voi.shape[1]
                    * self._Network__v_size
                    * random.random(),
                    self._Network__voi.shape[2]
                    * self._Network__v_size
                    * random.random(),
                )
            )
            max_length = (
                math.sqrt(
                    self._Network__voi.shape[0] ** 2
                    + self._Network__voi.shape[1] ** 2
                    + self._Network__voi.shape[2] ** 2
                )
                * self._Network__v_size
            )
            p_len = self.__gen_hfib_params.gen_persistence_length(
                self.__min_p_len
            )
            z_len_f = self.__gen_hfib_params.gen_zf_length(self.__mz_len_f)
            hold_polymer = HelixFiber(
                self.__l_length,
                self.__m_surf,
                p_len,
                self.__hp_len,
                self.__mz_len,
                z_len_f,
                p0,
            )

            # Polymer loop
            not_finished = True
            while (hold_polymer.get_total_len() < max_length) and not_finished:
                monomer_data = hold_polymer.gen_new_monomer(
                    self.__over_tolerance,
                    self._Network__voi,
                    self._Network__v_size,
                    net=self,
                    max_dist=self.__unit_diam,
                )
                if monomer_data is None:
                    not_finished = False
                else:
                    new_len = points_distance(
                        monomer_data[0], hold_polymer.get_tail_point()
                    )
                    if hold_polymer.get_total_len() + new_len < max_length:
                        hold_polymer.add_monomer(
                            monomer_data[0],
                            monomer_data[1],
                            monomer_data[2],
                            monomer_data[3],
                        )
                    else:
                        not_finished = False

            # Updating polymer
            if hold_polymer.get_num_mmers() >= self._Network__min_nmmer:
                self.add_polymer(hold_polymer)
                # print('build_network: new polymer added with ' + str(hold_polymer.get_num_monomers()) +
                #       ' and length ' + str(hold_polymer.get_total_len()) + ': occupancy ' + str(self._Network__pl_occ))


class NetHelixFiberB(Network):
    """
    Class for generating a network of brancked helix fibers randomly distributed
    """

    def __init__(
        self,
        voi,
        v_size,
        l_length,
        m_surf,
        gen_hfib_params,
        occ,
        min_p_len,
        hp_len,
        mz_len,
        mz_len_f,
        b_prop,
        max_p_branch=0,
        over_tolerance=0,
    ):
        """
        Construction

        :param voi: a 3D numpy array to define a VOI (Volume Of Interest) for polymers
        :param v_size: voxel size (default 1)
        :param l_length: polymer link length
        :param m_surf: monomer surf
        :param gen_hfib_params: a instance of a random generation model (random.PGen.NetHelixFiberB) to obtain random fiber
        parametrization for helix with branches
        :param occ: occupancy threshold in percentage [0, 100]%
        :param min_p_len: minimum persistence length
        :param hp_len: helix period length
        :param mz_len: monomer length in z-axis
        :param mz_len_f: maximum length factor in z-axis
        :param b_prob: branching probability, checked every time a new monomer is added
        :param max_p_branch: maximum number of branches per polymer, if 0 (default) then no branches are generated
        :param over_tolerance: fraction of overlapping tolerance for self avoiding (default 0, in range [0,1))
        """

        # Initialize abstract variables
        super(NetHelixFiberB, self).__init__(voi, v_size)

        # Input parsing
        assert l_length > 0
        assert isinstance(m_surf, vtk.vtkPolyData)
        assert isinstance(gen_hfib_params, HxParamGenBranched)
        assert (occ >= 0) and (occ <= 100)
        assert (over_tolerance >= 0) and (over_tolerance <= 100)
        assert (
            (min_p_len > 0)
            and (hp_len > 0)
            and (mz_len > 0)
            and (mz_len_f >= 0)
        )
        assert (max_p_branch >= 0) and (b_prop >= 0)

        # Variables assignment
        self.__l_length, self.__m_surf = l_length, m_surf
        self.__gen_hfib_params = gen_hfib_params
        self.__occ, self.__over_tolerance = occ, over_tolerance
        self.__min_p_len, self.__hp_len = min_p_len, hp_len
        self.__mz_len, self.__mz_len_f = mz_len, mz_len_f
        self.__max_p_branch, self.__p_branches, self.__b_prop = (
            max_p_branch,
            list(),
            b_prop,
        )

    def build_network(self):
        """
        Add helix fibres until an occupancy limit is passed

        :return:
        """

        MAX_TRIES = 1000
        tries_count = 0

        # Network loop
        while self._Network__pl_occ < self.__occ and tries_count < MAX_TRIES:
            tries_count += 1

            # Polymer initialization
            max_length = (
                math.sqrt(
                    self._Network__voi.shape[0] ** 2
                    + self._Network__voi.shape[1] ** 2
                    + self._Network__voi.shape[2] ** 2
                )
                * self._Network__v_size
            )
            p_len = self.__gen_hfib_params.gen_persistence_length(
                self.__min_p_len
            )
            z_len_f = self.__gen_hfib_params.gen_zf_length(self.__mz_len_f)
            branch = None
            if (self.__max_p_branch > 0) and self.__gen_hfib_params.gen_branch(
                self.__b_prop
            ):
                branch = self.__gen_random_branch()
            if branch is None:
                p0 = np.asarray(
                    (
                        self._Network__voi.shape[0]
                        * self._Network__v_size
                        * random.random(),
                        self._Network__voi.shape[1]
                        * self._Network__v_size
                        * random.random(),
                        self._Network__voi.shape[2]
                        * self._Network__v_size
                        * random.random(),
                    )
                )
            else:
                p0 = branch.get_point()
            hold_polymer = HelixFiber(
                self.__l_length,
                self.__m_surf,
                p_len,
                self.__hp_len,
                self.__mz_len,
                z_len_f,
                p0,
            )

            # Polymer loop
            not_finished = True
            while (hold_polymer.get_total_len() < max_length) and not_finished:
                monomer_data = hold_polymer.gen_new_monomer(
                    self.__over_tolerance,
                    self._Network__voi,
                    self._Network__v_size,
                )
                if monomer_data is None:
                    not_finished = False
                else:
                    new_len = points_distance(
                        monomer_data[0], hold_polymer.get_tail_point()
                    )
                    if hold_polymer.get_total_len() + new_len < max_length:
                        hold_polymer.add_monomer(
                            monomer_data[0],
                            monomer_data[1],
                            monomer_data[2],
                            monomer_data[3],
                        )
                    else:
                        not_finished = False

            # Updating polymer
            if hold_polymer.get_num_mmers() >= self._Network__min_nmmer:
                if branch is not None:
                    self.add_polymer(hold_polymer)
                    self.__p_branches.append(list())
                    self.__add_branch(hold_polymer, branch)
                else:
                    self.add_polymer(hold_polymer)
                    self.__p_branches.append(list())
                # print('build_network: new polymer added with ' + str(hold_polymer.get_num_monomers()) +
                #       ' and length ' + str(hold_polymer.get_total_len()) + ': occupancy ' + str(self._Network__pl_occ))

    def get_branch_list(self):
        """
        Get all branches in a list

        :return: a single list with the branches
        """
        hold_list = list()
        for bl in self.__p_branches:
            for b in bl:
                hold_list.append(b)
        return hold_list

    def get_skel(self):
        """
        Get Polymers Network as a vtkPolyData as points and lines with branches

        :return: a vtkPolyData
        """
        if len(self._Network__pl) == 0:
            return vtk.vtkPolyData()

        # Initialization
        app_flt_l, app_flt_v, app_flt = (
            vtk.vtkAppendPolyData(),
            vtk.vtkAppendPolyData(),
            vtk.vtkAppendPolyData(),
        )

        # Polymers loop
        p_type_l = vtk.vtkIntArray()
        p_type_l.SetName(NET_TYPE_STR)
        p_type_l.SetNumberOfComponents(1)
        for pol in self._Network__pl:
            app_flt_l.AddInputData(pol.get_skel())
        app_flt_l.Update()
        out_vtp_l = app_flt_l.GetOutput()
        for i in range(out_vtp_l.GetNumberOfCells()):
            p_type_l.InsertNextTuple((1,))
        out_vtp_l.GetCellData().AddArray(p_type_l)

        # Branches loop
        p_type_v = vtk.vtkIntArray()
        p_type_v.SetName(NET_TYPE_STR)
        p_type_v.SetNumberOfComponents(1)
        for i, branch in enumerate(self.get_branch_list()):
            app_flt_v.AddInputData(branch.get_vtp())
            # print('Point ' + str(i) + ': ' + str(branch.get_point()))
        app_flt_v.Update()
        out_vtp_v = app_flt_v.GetOutput()
        for i in range(out_vtp_v.GetNumberOfCells()):
            p_type_v.InsertNextTuple((2,))
        out_vtp_v.GetCellData().AddArray(p_type_v)

        # Merging branches and polymers
        app_flt.AddInputData(out_vtp_l)
        app_flt.AddInputData(out_vtp_v)
        app_flt.Update()

        return app_flt.GetOutput()

    def get_branches_vtp(self, shape_vtp=None):
        """
        Get Branches as a vtkPolyData with points

        :param shape_vtp: if None (default) the a point is returned, otherwise this shape is used
                          TODO: so far only isotropic shapes are recommended and starting monomer tangent is not considered yet
        :return: a vtkPolyData
        """

        # Initialization
        app_flt_l, app_flt_v, app_flt = (
            vtk.vtkAppendPolyData(),
            vtk.vtkAppendPolyData(),
            vtk.vtkAppendPolyData(),
        )

        # Branches loop
        for i, branch in enumerate(self.get_branch_list()):
            app_flt_v.AddInputData(branch.get_vtp(shape_vtp))
            # print('Point ' + str(i) + ': ' + str(branch.get_point()))
        app_flt_v.Update()
        out_vtp_v = app_flt_v.GetOutput()

        # Merging branches and polymers
        app_flt.AddInputData(out_vtp_v)
        app_flt.Update()

        return app_flt.GetOutput()

    def __gen_random_branch(self):
        """
        Generates a position point randomly for a branch on the filament network, no more than one branch per polymer

        :return: a branch
        """

        # Loop for polymers
        count, branch = 0, None
        while (count < len(self._Network__pl)) and (branch is None):
            hold_pid = random.choices(
                range(0, len(self._Network__pl)),
                weights=self._Network__pl_nmmers,
            )[0]
            if len(self.__p_branches[hold_pid]) < self.__max_p_branch:
                hold_pol = self._Network__pl[hold_pid]
                hold_mid = random.randint(0, len(hold_pol._Polymer__m) - 1)
                hold_m = hold_pol._Polymer__m[hold_mid]
                found = True
                for branch in self.__p_branches[hold_pid]:
                    if (
                        points_distance(
                            hold_m.get_center_mass(), branch.get_point()
                        )
                        <= 2 * hold_m.get_diameter()
                    ):
                        found = False
                if found:
                    branch = Branch(
                        hold_m.get_center_mass(), hold_pid, hold_mid
                    )
            count += 1

        return branch

    def __add_branch(self, polymer, branch):
        """
        Add a new branch to the polymer network

        :param polymer: targeting polymer where the branch is going to be added (starting polumer is obtained from
                        the branch)
        :param branch: branch to be added
        """
        branch.set_t_pmer(len(self._Network__pl) - 1)
        self.__p_branches[branch.get_s_pmer()].append(branch)

@dataclass
class Branch:
    """
    Class to model a branch in a Network
    """
    point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    s_pmer_id: int = 0
    s_mmer_id: int = 0
    t_pmer_id: int = None

    def __post_init__(self):
        assert hasattr(self.point, "__len__") and (len(self.point) == 3)
        self.point = np.asarray(self.point, dtype=float)
        assert (self.s_pmer_id >= 0) and (self.s_mmer_id >= 0)
        if self.t_pmer_id is not None:
            assert self.t_pmer_id >= 0

    def set_t_pmer(self, t_pmer_id):
        """
        Set targeting polymer ID
        """
        assert t_pmer_id >= 0
        self.t_pmer_id = t_pmer_id

    def get_point(self):
        """
        Get point coordinates
        """
        return self.point

    def get_s_pmer(self):
        """
        Get starting polymer ID
        """
        return self.s_pmer_id

    def get_s_mmer(self):
        """
        Get starting monomer ID
        """
        return self.s_mmer_id

    def get_t_pmer(self):
        """
        Get targeting polymer ID
        """
        return self.t_pmer_id

    def get_vtp(self, shape_vtp=None):
        """
        Gets a polydata with the branch shape

        :param shape_vtp: if None (default) the a point is returned, otherwise this shape is used
                          TODO: so far only isotropic shapes are recommended and starting monomer tangent is not considered yet
        :return:
        """
        if shape_vtp is None:
            return point_to_poly(self.get_point())
        else:
            return poly_translate(shape_vtp, self.get_point())
