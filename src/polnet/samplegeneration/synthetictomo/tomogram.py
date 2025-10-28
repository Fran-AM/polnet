"""Module for Subvolume and Tomogram class definition. """
from abc import ABC
from pprint import pp
from time import time
import numpy as np

from polnet.samplegeneration.membranes.mb_factory import MbFactory
from polnet.samplegeneration.membranes.mb_generator import MbGen
from polnet.samplegeneration.membranes.mb_set import SetMembranes
from polnet.utils import poly as pp

class Tomogram:
    """
    This class model a Tomogram
    """

    LBL_MB = 1  # Label for membranes in the polydata

    # TODO: initialize from density array or from file
    def __init__(self, id, shape, v_size, offset=(4,4,4)):
        self.__id = id
        self.__shape = shape
        self.__v_size = v_size

        self.__voi = np.zeros(shape=self.__shape, dtype=bool)
        self.__voi[
            offset[0]: self.__shape[0] - offset[0],
            offset[1]: self.__shape[1] - offset[1],
            offset[2]: self.__shape[2] - offset[2]
        ] = True
        self.__voi_voxels = self.__voi.sum()
        self.__bg_voi = self.__voi.copy()
        self.__labels = np.zeros(shape=self.__shape, dtype=np.int16)
        self.__density = np.zeros(shape=self.__shape, dtype=np.float32)  
        self.__poly_vtp, self.__mbs_vtp, self.__skel_vtp = None, None, None
        self.__entity_id = 1
        self.__structure_counts = {
            'membrane': 0,
        }
        self.__voxel_counts = {
            'membrane': 0,
        }

    @property
    def id(self):
        """Get the tomogram ID.

        Returns:
            int: The tomogram ID.
        """
        return self.__id

    @property
    def density(self):
        """Get the tomogram density.

        Returns:
            np.ndarray: The tomogram density.
        """
        return self.__density
    
    @property
    def labels(self):
        """Get the tomogram labels.

        Returns:
            np.ndarray: The tomogram labels.
        """
        return self.__labels
    
    @property
    def voi(self):
        """Get the volume of interest.

        Returns:
            np.ndarray: The volume of interest.
        """
        return self.__voi
    
    @property
    def volume(self):
        """Get the volume of the tomogram in cubic angstroms.

        Returns:
            float: The volume in cubic angstroms.
        """
        return float(self.__voi_voxels) * self.__v_size * self.__v_size * self.__v_size
    
    def voxel_counts(self, type: str):
        """Get the voxel counts for a given type.

        Args:
            type (str): The type of voxel count to retrieve ('membrane').

        Returns:
            int: The voxel count for the specified type.
        """
        assert type in self.__voxel_counts, f"Invalid voxel count type: {type}. Valid types are: {list(self.__voxel_counts.keys())}"
        return self.__voxel_counts.get(type, 0)
    
    def structure_counts(self, type: str):
        """Get the structure counts for a given type.

        Args:
            type (str): The type of structure count to retrieve ('membrane', 'network', 'actin').

        Returns:
            int: The structure count for the specified type.
        """
        assert type in self.__structure_counts, f"Invalid structure count type: {type}. Valid types are: {list(self.__structure_counts.keys())}"
        return self.__structure_counts.get(type, 0)

    def gen_set_mbs(self, params: dict, cf: float = None, verbosity: bool = True):
        """Generate a set of membranes and add them to the tomogram.

        Args:
            params (dict): The parameters for the membrane generator.
            cf (float, optional): The contrast factor to apply. If None, no contrast factor is applied. Defaults to None.
            verbosity (bool, optional): Whether to print progress messages. Defaults to True.

        Raises:
            ValueError: If 'MB_TYPE' is not in params.    

        Returns:
            None
        """

        # Retrieving membrane generator from parameters
        if "MB_TYPE" not in params:
            raise ValueError("Missing 'MB_TYPE' in parameters")

        mb_type = params["MB_TYPE"]
        mb_generator = MbFactory.create(mb_type, params)
     
        # Creating the set of membranes
        set_mbs = SetMembranes(
            voi=self.__voi,
            v_size=self.__v_size,
            gen_rnd_surfs=mb_generator,
            bg_voi=self.__bg_voi,
        )

        # Building the set of membranes
        set_mbs.build_set(verbosity=verbosity)
        hold_den = set_mbs.density
        if cf is not None:
            hold_den = hold_den * cf
        hold_mask = set_mbs.gtruth
        hold_vtp = set_mbs.vtp

        # Update tomogram voi, density and labels
        self.__voi = self.__voi & (~hold_mask)
        self.__density = np.maximum(self.__density, hold_den)
        self.__labels[hold_mask] = self.__entity_id
        self.__voxel_counts['membrane'] += hold_mask.sum()
        self.__structure_counts['membrane'] += 1

        # Update tomogram polydata
        pp.add_label_to_poly(hold_vtp, self.__entity_id, "Entity", mode="both")
        pp.add_label_to_poly(hold_vtp, self.LBL_MB, "Type", mode="both")
        if self.__poly_vtp is None:
            self.__poly_vtp = hold_vtp
            self.__skel_vtp = hold_vtp
        else:
            self.__poly_vtp = pp.merge_polys(self.__poly_vtp, hold_vtp)
            self.__skel_vtp = pp.merge_polys(self.__skel_vtp, hold_vtp)
        self.__entity_id += 1

