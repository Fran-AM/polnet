import random

import numpy as np

import polnet.utils.poly as pp

class PnSet:
    """
    Class for generating a set of cytosolic protein entities within a volume of interest
    """

    def __init__(
        self,
        voi,
        bg_voi,
        v_size,
        gen_rnd_cproteins,
        surf_dec: float = 0.9,
        tries_mmer: float = 20,
        tries_pmer: float = 100,
        verbosity: bool = True
    ):
        self.__voi = voi
        self.__bg_voi = bg_voi
        self.__v_size = v_size
        self.__gen_rnd_cproteins = gen_rnd_cproteins
        self.__surf_dec = surf_dec
        self.__verbosity = verbosity
        self.__tries_mmer = tries_mmer
        self.__tries_pmer = tries_pmer
        self.__occ = 0.0

    def build_set(self):
        """Build the set of cytosolic protein entities within the volume of interest.

        Raises:
            NotImplementedError: Cytosolic protein set building not yet implemented.
        """

        # Get surface model and scale to voxel size
        surf = pp.poly_scale(self.__gen_rnd_cproteins.surf, self.__v_size)
        surf_diam = pp.poly_diam(surf)

        c_try = 0
        pmer_fails = 0
        max_occ = self.__gen_rnd_cproteins.rnd_occ()

        # Network loop
        while (c_try <= self.__tries_pmer) and (self.__occ < max_occ):
            # Polymer initailization
            c_try += 1
            p0 = np.asarray(
                (
                    self.__voi.shape[0] * self.__v_size * random.random(),
                    self.__voi.shape[1] * self.__v_size * random.random(),
                    self.__voi.shape[2] * self.__v_size * random.random(),
                )
            )
            max_length = random.uniform(0, self.__gen_rnd_cproteins.pmer_l * surf_diam)


        