"""Module for managing a synthetic tomogram"""

from polnet.samplegeneration.membranes.mb import Mb
import polnet.utils.poly as pp

class SynthTomo:
    """
    This class model a Synthetic tomogram, tomogram's info are stored in disk to handle large datasets:
        - Ground truth: integer type tomogram associating each pixel with the id (background is always 0)
    """

    def __init__(self):
        self.__den = None
        self.__tomo = None
        self.__mics = None
        self.__poly = None
        self.__motifs = list()

    def get_den(self):
        return self.__den

    def get_mics(self):
        return self.__mics

    def get_poly(self):
        return self.__poly

    def get_tomo(self):
        return self.__tomo

    def set_den(self, den):
        assert isinstance(den, str) and den.endswith(".mrc")
        self.__den = den

    def set_tomo(self, tomo):
        assert isinstance(tomo, str) and tomo.endswith(".mrc")
        self.__tomo = tomo

    def set_mics(self, mics):
        assert isinstance(mics, str) and mics.endswith(".mrc")
        self.__mics = mics

    def set_poly(self, poly):
        assert isinstance(poly, str) and poly.endswith(".vtp")
        self.__poly = poly

    def get_motif_list(self):
        return self.__motifs

    # def add_network(self, net, m_type, lbl, code=None):
    #     """
    #     Add all motifs within the input network to the synthetic tomogram

    #     :param net: a network object instance
    #     :param m_type: string with the type of monomers contained in the network
    #     :param lbl: integer label
    #     :param code: string code for the network monomers, if None (default) it is taken from monomer information
    #     """
    #     assert issubclass(type(net), Network)
    #     assert isinstance(m_type, str)
    #     if lbl is not None:
    #         assert isinstance(lbl, int)
    #     if code is not None:
    #         assert isinstance(code, str)
    #     for pmer_id, pmer in enumerate(net.get_pmers_list()):
    #         for mmer_id in range(pmer.get_num_monomers()):
    #             if code is None:
    #                 hold_code = pmer.get_mmer_code(mmer_id)
    #             else:
    #                 hold_code = code
    #             self.__motifs.append(
    #                 list(
    #                     (
    #                         m_type,
    #                         lbl,
    #                         hold_code,
    #                         pmer_id,
    #                         pmer.get_mmer_center(mmer_id),
    #                         pmer.get_mmer_rotation(mmer_id),
    #                     )
    #                 )
    #             )

    # def add_set_mbs(self, set_mbs, m_type, lbl, code, dec=None):
    #     """
    #     Membrane surface point coordinates are added to the tomogram motif list
    #     In rotations the normal vector to each point is stored as: X->Q0, Y->Q1 , Z->Q2 and 0->Q3

    #     :param set_mbs: a membrane set object instance
    #     :param m_type: string with the type of motif contained in the network
    #     :param lbl: integer label
    #     :param code: string code for membrane
    #     :param dec: if not None (default) the membrane points are decimated according this factor
    #     """
    #     assert issubclass(type(set_mbs), SetMembranes)
    #     assert isinstance(m_type, str)
    #     assert isinstance(lbl, int)
    #     assert isinstance(code, str)

    #     poly_vtp = set_mbs.get_vtp()
    #     if dec is not None:
    #         poly_vtp = pp.poly_decimate(poly_vtp, dec)

    #     n_points = poly_vtp.GetNumberOfPoints()
    #     normals = poly_vtp.GetPointData().GetNormals()
    #     for i in range(n_points):
    #         x, y, z = poly_vtp.GetPoint(i)
    #         q0, q1, q2 = normals.GetTuple(i)
    #         self.__motifs.append(
    #             list((m_type, lbl, code, i, [x, y, z], [q0, q1, q2, 0]))
    #         )

    # TODO: provisional until set membranes is fixed
    def add_mbs(self, mb, lbl, code, dec=None):
        """
        Membrane surface point coordinates are added to the tomogram motif list
        In rotations the normal vector to each point is stored as: X->Q0, Y->Q1 , Z->Q2 and 0->Q3

        :param mb: a membrane object instance
        :param lbl: integer label
        :param code: string code for membrane
        :param dec: if not None (default) the membrane points are decimated according this factor
        """
        assert issubclass(type(mb), Mb)
        assert isinstance(lbl, int)
        assert isinstance(code, str)

        poly_vtp = mb.vtp
        if dec is not None:
            poly_vtp = pp.poly_decimate(poly_vtp, dec)

        n_points = poly_vtp.GetNumberOfPoints()
        normals = poly_vtp.GetPointData().GetNormals()
        for i in range(n_points):
            x, y, z = poly_vtp.GetPoint(i)
            q0, q1, q2 = normals.GetTuple(i)
            self.__motifs.append(list((mb.get_type(), lbl, code, i, [x, y, z], [q0, q1, q2, 0])))

    def add_offset(self, offset: list) -> None:
        """
        Add an offset to the coordinates of the particles in the motif list

        Args:
            offset (list): A list of 3 elements representing the offset in X, Y, and Z directions

        Returns:
            None

        """
        for i in range(len(self.__motifs)):
            self.__motifs[i][5][0] = offset[0]
            self.__motifs[i][5][1] = offset[1]
            self.__motifs[i][5][2] = offset[2]