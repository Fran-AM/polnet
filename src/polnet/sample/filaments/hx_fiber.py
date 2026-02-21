from ..polymers import Polymer, Monomer
import math
import numpy as np
from ...utils.utils import (
    points_distance,
    gen_uni_s2_sample
)

from ...utils.affine import (
    angle_axis_to_quat,
    vector_module,
    rot_vect_quat,
    rot_to_quat,
    quat_mult,
    vect_to_zmat,
    wrap_angle
)


class HelixFiber(Polymer):
    """
    Class for modelling a random helical flexible fiber
    """

    def __init__(
        self,
        l_length,
        m_surf,
        p_length,
        hp_length,
        mz_length,
        z_length_f=0,
        p0=(0, 0, 0),
        vz=(0, 0, 1),
        rot_rand=True,
    ):
        """
        Constructor

        :param l_length: link length
        :param m_surf: monomer surface (as vtkPolyData object)
        :param p_length: persistence length
        :param hp_length: helix period length (distance required by azimuthal angle to cover 360deg)
        :param mz_length: monomer z-length
        :param z_length_f: helix elevation factor or slope
        :param p0: starting point (default origin (0,0,0))
        :param vz: reference vector for z-axis (default (0, 0, 1)
        :param rot_rand: if True (default) the rotation of the first monomer (and consequently its tangent) is
                         generated randomly, otherwise it is set to fit vz
        """
        super(HelixFiber, self).__init__(m_surf)
        assert (
            (l_length > 0)
            and (p_length > 0)
            and (z_length_f >= 0)
            and (hp_length > 0)
            and (mz_length > 0)
        )
        self.__l, self.__lp, self.__lz = l_length, p_length, z_length_f
        self.__hp, self.__mz_length = hp_length, mz_length
        self.__hp_astep = (360.0 * self.__mz_length) / self.__hp
        self.__compute_helical_parameters()
        assert hasattr(vz, "__len__") and (len(vz) == 3)
        self.__vz = np.asarray(vz, dtype=float)
        # Curve state member variables
        self.__ct, self.__za, self.__rq = (
            0.0,
            0.0,
            np.asarray((1.0, 0.0, 0.0, 0.0)),
        )  # z-aligned curve time (considering speed 1)
        self.set_reference(np.asarray(p0), self.__vz, rot_rand=rot_rand)

    def set_reference(
        self, p0=(0.0, 0.0, 0.0), vz=(0.0, 0.0, 1.0), rot_rand=True
    ):
        """
        Initializes the chain with the specified input point, if points are introduced before they will be forgotten

        :param p0: starting point
        :param vz: z-axis reference vector for helicoid parametrization
        :param rot_rand: if True (default) the rotation of the first monomer (and consequently its tangent) is
                         generated randomly, otherwise it is set to fit vz
        :return:
        """
        assert hasattr(p0, "__len__") and (len(p0) == 3)
        self._Polymer__p = np.asarray(p0)
        vzr = vz / vector_module(vz)
        if rot_rand:
            t = gen_uni_s2_sample(np.asarray((0.0, 0.0, 0.0)), 1.0)
            M = vect_to_zmat(t, mode="passive")
            self.__rq = rot_to_quat(M)
        else:
            self.__rq = np.asarray((1.0, 0.0, 0.0, 0.0))
        t = self.__compute_tangent(self.__ct)
        t = t * (self.__mz_length / vector_module(t))
        self.__ct += self.__l
        q1 = angle_axis_to_quat(self.__za, t[0], t[1], t[2])
        M = vect_to_zmat(t, mode="passive")
        q = rot_to_quat(M)
        hold_q = quat_mult(q, q1)
        # vzr *= self.__mz_length
        hold_monomer = Monomer(self._Polymer__m_surf, self._Polymer__m_diam)
        hold_monomer.rotate_q(hold_q)
        hold_monomer.translate(p0)
        # self.__rq = hold_q
        self.add_monomer(p0, t, hold_q, hold_monomer)

    def gen_new_monomer(
        self,
        over_tolerance=0,
        voi=None,
        v_size=1,
        net=None,
        branch=None,
        max_dist=None,
    ):
        """
        Generates a new monomer according the flexible fiber model

        :param over_tolerance: fraction of overlapping tolerance for self avoiding (default 0)
        :param voi: VOI to define forbidden regions (default None, not applied)
        :param v_size: VOI voxel size, it must be greater than 0 (default 1)
        :param net: if not None (default) it contain a network of polymers that must be avoided
        :param branch: input branch from where the current mmer starts, is avoid network avoiding at the branch,
                       only valid in net is not None (default None).
        :param max_dist: allows to externally set a maximum distance (in A) to search for collisions for network
                         overlapping, otherwise 1.2 monomer diameter is used
        :return: a 4-tuple with monomer center point, associated tangent vector, rotated quaternion and monomer,
                 return None in case the generation has failed
        """

        hold_m = Monomer(self._Polymer__m_surf, self._Polymer__m_diam)

        # Rotation
        t = self.__compute_tangent(self.__ct)
        t = t * (self.__mz_length / vector_module(t))
        self.__za = wrap_angle(self.__za + self.__hp_astep)
        q1 = angle_axis_to_quat(self.__za, t[0], t[1], t[2])
        M = vect_to_zmat(t, mode="passive")
        q = rot_to_quat(M)
        hold_m.rotate_q(quat_mult(q, q1))

        # Translation
        hold_r = self._Polymer__r[-1]
        self.__ct += self.__l
        r = hold_r + t
        hold_m.translate(r)

        # Avoid forbidden regions
        if voi is not None:
            if hold_m.overlap_voi(voi, v_size, over_tolerance=over_tolerance):
                return None
        # Self-avoiding and network avoiding
        if branch is None:
            if self.overlap_polymer(hold_m, over_tolerance=over_tolerance):
                return None
            if net is not None:
                if hold_m.overlap_net(
                    net, over_tolerance=over_tolerance, max_dist=max_dist
                ):
                    return None
        else:
            branch_dst = points_distance(
                branch.get_point(), hold_m.get_center_mass()
            )
            if branch_dst > hold_m.get_diameter():
                if self.overlap_polymer(hold_m, over_tolerance=over_tolerance):
                    return None
                if net is not None:
                    if hold_m.overlap_net(net, over_tolerance=over_tolerance):
                        return None

        return r, t, q, hold_m

    def __compute_helical_parameters(self):
        """
        Private method (fill class member variables) to compute helical parameters (a and b) from persistence length

        :return:
        """

        # Compute curvature from persistence
        k = math.acos(math.exp(-self.__l / self.__lp)) / self.__l

        # Compute circular parameter, a, from curvature
        self.__a = 1 / k

        # Compute Z-axis elevation from the circular parameter
        self.__b = self.__lz * self.__a

    def __compute_tangent(self, t):
        """
        Computes curve (z-aligned axis) normalized tangent vector

        :param t: input parameter, time assuming that speed is 1
        :return: returns the normalized tangent vector (3 elements array)
        """
        sq = math.sqrt(self.__a * self.__a + self.__b * self.__b)
        s = t
        s_sq = s / sq
        t = (1.0 / sq) * np.asarray(
            (-self.__a * math.sin(s_sq), self.__a * math.cos(s_sq), self.__b)
        )
        return rot_vect_quat(t, self.__rq)