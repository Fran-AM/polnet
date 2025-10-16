"""Module for managing membrane configuration files"""

class MbFile:
    """
    For handling membrane configuration files
    """

    def __init__(self):
        self.__type = None
        self.__occ = None
        self.__thick_rg = None
        self.__layer_s_rg = None
        self.__max_ecc = None
        self.__over_tol = None
        self.__min_rad = None
        self.__max_rad = None
        self.__den_cf_rg = None

    def get_type(self):
        return self.__type

    def get_occ(self):
        return self.__occ

    def get_thick_rg(self):
        return self.__thick_rg

    def get_layer_s_rg(self):
        return self.__layer_s_rg

    def get_max_ecc(self):
        return self.__max_ecc

    def get_over_tol(self):
        return self.__over_tol

    def get_min_rad(self):
        return self.__min_rad

    def get_max_rad(self):
        return self.__max_rad

    def get_den_cf_rg(self):
        return self.__den_cf_rg

    def load_mb_file(self, in_file):
        """
        Load protein parameters from an input file

        :param in_file: path to the input file with extension .mbs
        """

        assert isinstance(in_file, str) and in_file.endswith(".mbs")

        # Reading input file
        with open(in_file) as file:
            for linea in file:
                if len(linea.strip()) > 0:

                    # Remove coments
                    linea = linea.split("#", 1)[0]

                    # Parsing an file entry
                    var, value = linea.split("=")
                    var = var.replace(" ", "")
                    var = var.replace("\n", "")
                    value = value.replace(" ", "")
                    value = value.replace("\n", "")
                    if var == "MB_TYPE":
                        self.__type = value
                    elif var == "MB_OCC":
                        try:
                            self.__occ = float(value)
                        except ValueError:
                            value_0 = value[
                                value.index("(") + 1 : value.index(",")
                            ]
                            value_1 = value[
                                value.index(",") + 1 : value.index(")")
                            ]
                            self.__occ = (float(value_0), float(value_1))
                    elif var == "MB_THICK_RG":
                        value_0 = value[value.index("(") + 1 : value.index(",")]
                        value_1 = value[value.index(",") + 1 : value.index(")")]
                        self.__thick_rg = (float(value_0), float(value_1))
                    elif var == "MB_LAYER_S_RG":
                        value_0 = value[value.index("(") + 1 : value.index(",")]
                        value_1 = value[value.index(",") + 1 : value.index(")")]
                        self.__layer_s_rg = (float(value_0), float(value_1))
                    elif var == "MB_MAX_ECC":
                        self.__max_ecc = float(value)
                    elif var == "MB_OVER_TOL":
                        self.__over_tol = float(value)
                    elif var == "MB_MIN_RAD":
                        self.__min_rad = float(value)
                    elif var == "MB_MAX_RAD":
                        self.__max_rad = float(value)
                    elif var == "MB_DEN_CF_RG":
                        value_0 = value[value.index("(") + 1 : value.index(",")]
                        value_1 = value[value.index(",") + 1 : value.index(")")]
                        self.__den_cf_rg = (float(value_0), float(value_1))
                    else:
                        print(
                            "ERROR: (MbFile - load_mb_file) input entry not recognized:",
                            value,
                        )
