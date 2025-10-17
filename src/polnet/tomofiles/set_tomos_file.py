"""Module for managing a set of simulated tomograms"""

import csv

from polnet.samplegeneration.synthetictomo import SynthTomo


class SetTomos:
    """Class for storing information for a set of simulated tomograms"""

    def __init__(self) -> None:
        """Constructor

        Initializes an empty list of tomograms

        Args:
            None

        Returns:
            None
        """
        self.__tomos = list()

    def add_tomos(self, tomo: SynthTomo) -> None:
        """Add a SynthTomo to the list

        :param tomo: input SyntTomo
        """
        assert isinstance(tomo, SynthTomo)
        self.__tomos.append(tomo)

    def save_csv(self, out_file):
        """
        Saves the motifs list contained in the set of tomograms

        :param out_file: output file path in .csv format
        """
        assert isinstance(out_file, str) and out_file.endswith(".csv")

        # Writing output CSV file
        with open(out_file, "w") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                dialect=csv.excel_tab,
                fieldnames=[
                    "Density",
                    "Micrographs",
                    "PolyData",
                    "Tomo3D",
                    "Type",
                    "Label",
                    "Code",
                    "Polymer",
                    "X",
                    "Y",
                    "Z",
                    "Q1",
                    "Q2",
                    "Q3",
                    "Q4",
                ],
            )
            writer.writeheader()

            # Tomos loop
            for tomo in self.__tomos:
                den_path = tomo.get_den()
                mics_path = tomo.get_mics()
                poly_path = tomo.get_poly()
                rec_path = tomo.get_tomo()

                # Motifs loop
                for motif in tomo.get_motif_list():
                    m_type = motif[0]
                    lbl_id = motif[1]
                    text_id = motif[2]
                    pmer_id = motif[3]
                    center = motif[4]
                    rotation = motif[5]

                    # Writing entry
                    writer.writerow(
                        {
                            "Density": den_path,
                            "Micrographs": mics_path,
                            "PolyData": poly_path,
                            "Tomo3D": rec_path,
                            "Type": m_type,
                            "Label": lbl_id,
                            "Code": text_id,
                            "Polymer": pmer_id,
                            "X": center[0],
                            "Y": center[1],
                            "Z": center[2],
                            "Q1": rotation[0],
                            "Q2": rotation[1],
                            "Q3": rotation[2],
                            "Q4": rotation[3],
                        }
                    )
