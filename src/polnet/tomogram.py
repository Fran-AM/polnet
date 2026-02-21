import csv
import time
from pathlib import Path
import sys

from .sample import SyntheticSample, MbFile, PnFile, HxFile
#from .tem import TEM, TEMFile
from .utils import lio
from .logging_conf import _LOGGER as logger

class SynthTomo():

    def __init__(
        self,
        id: int,
        mbs_file_list: list,
        hns_file_list: list,
        pns_file_list: list,
        pms_file_list: list,
        #tem_file_path: Path
    ):
        if not isinstance(mbs_file_list, list) or not all(isinstance(f, str) for f in mbs_file_list):
            raise TypeError("mbs_file_list must be a list of strings.")
        if not isinstance(hns_file_list, list) or not all(isinstance(f, str) for f in hns_file_list):
            raise TypeError("hns_file_list must be a list of strings")
        if not isinstance(pns_file_list, list) or not all(isinstance(f, str) for f in pns_file_list):
            raise TypeError("pns_file_list must be a list of strings")
        if not isinstance(pms_file_list, list) or not all(isinstance(f, str) for f in pms_file_list):
            raise TypeError("pms_file_list must be a list of strings")
        # if not isinstance(tem_file_path, Path):
        #     raise TypeError("tem_file_path must be a Path object")

        self.__id = id
        self.__mbs_files = mbs_file_list
        self.__hns_files = hns_file_list
        self.__pns_files = pns_file_list
        self.__pms_files = pms_file_list
        #self.__tem_file_path = tem_file_path
        self.__sample = None
        self.__temic = None
    
    def gen_sample(
        self,
        data_path: Path,
        shape: tuple,
        v_size: float,
        offset: tuple
    ) -> None:
        """Generate a synthetic sample with membranes, host and parasite networks.

        Args:
            data_path (Path): Path to the data directory containing the model files.
            shape (tuple): Shape of the volume of interest (VOI) in voxels.
            v_size (float): Voxel size in Angstroms.
            offset (tuple): Offset of the VOI in voxels.
        Returns:
            None
        """
        if self.__sample is not None:
            raise RuntimeError("Sample has already been generated.")

        logger.info("Generating synthetic sample.")

        tic = time.time()

        self.__sample = SyntheticSample(
            shape=shape,
            v_size=v_size,
            offset=offset
        )

        logger.info("Adding membranes to the sample.")

        for mb_file_rpath in self.__mbs_files:
            mb_file_apath = data_path / mb_file_rpath
            mb_file = MbFile()
            mb_params = mb_file.load(mb_file_apath)

            self.__sample.add_set_membranes(
                params=mb_params,
                max_mbtries=10
            )

        logger.info("Membranes added. Adding helicoidal networks to the sample.")

        for hn_file_rpath in self.__hns_files:
            hn_file_apath = data_path / hn_file_rpath
            hn_file = HxFile()
            hn_params = hn_file.load(hn_file_apath)

            self.__sample.add_helicoidal_network(
                params=hn_params
            )

        logger.info("Helicoidal networks added. Adding cytosolic proteins to the sample.")

        for pn_file_rpath in self.__pns_files:
            pn_file_apath = data_path / pn_file_rpath
            pn_file = PnFile()
            pn_params = pn_file.load(pn_file_apath)

            self.__sample.add_set_cproteins(
                params=pn_params,
                data_path=data_path,
                surf_dec=0.9,
                mmer_tries=20,
                pmer_tries=100
            )


        # TODO: Add the rest of components

        toc = time.time()
        logger.info(f"Synthetic sample generated in {toc - tic:.2f} seconds.")

        return None
    
    def tem(
        self,
        data_path: Path,
        output_folder: Path,
    ):# TODO complete
        """Simulate TEM imaging of the synthetic sample.

        Returns:
            None
        """
        pass
        # if output_folder is None or not isinstance(output_folder, Path):
        #     raise TypeError("output_folder must be a Path object.")
        # output_folder.mkdir(parents=True, exist_ok=True)

        # if self.__sample is None:
        #     raise RuntimeError("Sample has not been generated yet.")
        
        # tem_file_apath = data_path / self.__tem_file_path
        # tem_file = TEMFile()
        # tem_params = tem_file.load(tem_file_apath)
        # self.__temic = TEM(tem_params)
        # self.__temic.simulate(
        #     vol=self.__sample.density,
        #     params = tem_params
        # )
    
    def save_tomo(
        self,
        output_folder: Path,
    ) ->  None:

        if output_folder is None or not isinstance(output_folder, Path):
            raise TypeError("output_folder must be a Path object.")
        output_folder.mkdir(parents=True, exist_ok=True)

        # Save labels table
        self.__save_labels_table(output_folder)

        # Save synthetic sample files
        den_path = output_folder / f"tomo_{self.__id:03d}_den.mrc"
        lio.write_mrc(
            self.__sample.density,
            den_path,
            v_size=self.__sample.v_size,
        )

        lbl_path = output_folder / f"tomo_{self.__id:03d}_lbl.mrc"
        lio.write_mrc(
            self.__sample.labels,
            lbl_path,
            v_size=self.__sample.v_size,
        )

        if self.__sample.poly_vtp is not None:
            poly_den_path = output_folder / f"tomo_{self.__id:03d}_poly_den.vtp"
            lio.save_vtp(
                self.__sample.poly_vtp,
                poly_den_path,
            )
        else:
            logger.warning("No poly_vtp data to save.")

        if self.__sample.skel_vtp is not None:
            poly_skel_path = output_folder / f"tomo_{self.__id:03d}_poly_skel.vtp"
            lio.save_vtp(
                self.__sample.skel_vtp,
                poly_skel_path,
            )
        else:
            logger.warning("No skel_vtp data to save.")
        
    def __save_labels_table(
            self,
            output_folder: Path,
        ) -> None:
            """Save the labels table to a CSV file.

            Args:
                out_file (Path): Path to the output CSV file.

            Returns:
                None
            """
            out_file = output_folder / "labels_table.csv"
            unit_lbl = 1
            header_lbl_tab = ["MODEL", "LABEL"]
            with open(out_file, "w") as file_csv:
                writer_csv = csv.DictWriter(
                    file_csv, fieldnames=header_lbl_tab, delimiter="\t"
                )
                writer_csv.writeheader()
                for fname in self.__mbs_files:
                    writer_csv.writerow(
                        {header_lbl_tab[0]: fname, header_lbl_tab[1]: unit_lbl}
                    )
                    unit_lbl += 1
                for fname in self.__hns_files:
                    writer_csv.writerow(
                        {header_lbl_tab[0]: fname, header_lbl_tab[1]: unit_lbl}
                    )
                    unit_lbl += 1
                for fname in self.__pns_files:
                    writer_csv.writerow(
                        {header_lbl_tab[0]: fname, header_lbl_tab[1]: unit_lbl}
                    )
                    unit_lbl += 1
                for fname in self.__pms_files:
                    writer_csv.writerow(
                        {header_lbl_tab[0]: fname, header_lbl_tab[1]: unit_lbl}
                    )
                    unit_lbl += 1

            return None

    def summary(self) -> None:
        """Show a summary of the synthetic sample.

        Returns:
            None
        """
        if self.__sample is None:
            logger.error("No sample generated yet.")
            return
        
        self.__sample.summary()
        
        return None