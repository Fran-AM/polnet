import math
import os
import sys
import time
import random

import numpy as np

from polnet.utils import poly as pp
from polnet.utils import lio 
from polnet.samplegeneration.synthetictomo.synth_tomo import SynthTomo
from polnet.samplegeneration.membranes.set_membranes import SetMembranes
from polnet.tomofiles.mb_file import MbFile

# Common tomogram settings
# ROOT_PATH should point to the 'data' folder
ROOT_PATH = os.path.realpath(os.getcwd() + "/../../data")
NTOMOS = 1
VOI_SHAPE = (
    300,
    300,
    250,
)
VOI_OFFS = (
    (4, VOI_SHAPE[0] - 4),
    (4, VOI_SHAPE[1] - 4),
    (4, VOI_SHAPE[2] - 4),
)
VOI_VSIZE = 10 # A/vx
MMER_TRIES = 50
PMER_TRIES = 10

# Lists with the features to simulate
MEMBRANES_LIST = [
    "in_mbs/sphere.mbs",
    # "in_mbs/ellipse.mbs",
    # "in_mbs/toroid.mbs",
]

# HELIX_LIST = [
#     "in_helix/mt.hns",
#     "in_helix/actin.hns"
#     ]

# PROTEINS_LIST = [
#     "in_10A/4v4r_10A.pns",
#     "in_10A/3j9i_10A.pns",
#     "in_10A/4v4r_50S_10A.pns",
#     "in_10A/4v4r_30S_10A.pns",
#     "in_10A/6utj_10A.pns",
#     "in_10A/5mrc_10A.pns",
#     "in_10A/4v7r_10A.pns",
#     "in_10A/2uv8_10A.pns",
#     "in_10A/4v94_10A.pns",
#     "in_10A/4cr2_10A.pns",
#     "in_10A/3qm1_10A.pns",
#     "in_10A/3h84_10A.pns",
#     "in_10A/3gl1_10A.pns",
#     "in_10A/3d2f_10A.pns",
#     "in_10A/3cf3_10A.pns",
#     "in_10A/2cg9_10A.pns",
#     "in_10A/1u6g_10A.pns",
#     "in_10A/1s3x_10A.pns",
#     "in_10A/1qvr_10A.pns",
#     "in_10A/1bxn_10A.pns",
# ]

# MB_PROTEINS_LIST = [
#     "in_10A/mb_6rd4_10A.pms",
#     "in_10A/mb_5wek_10A.pms",
#     "in_10A/mb_4pe5_10A.pms",
#     "in_10A/mb_5ide_10A.pms",
#     "in_10A/mb_5gjv_10A.pms",
#     "in_10A/mb_5kxi_10A.pms",
#     "in_10A/mb_5tj6_10A.pms",
#     "in_10A/mb_5tqq_10A.pms",
#     "in_10A/mb_5vai_10A.pms",
# ]

# Proportions list, specifies the proportion for each protein, this proportion is tried to be achieved but no guaranteed
# The total sum of this list must be 1
PROP_LIST = None  # [.4, .6]
if PROP_LIST is not None:
    assert sum(PROP_LIST) == 1

SURF_DEC = 0.9  # Target reduction factor for surface decimation (default None)

# Reconstruction tomograms
TILT_ANGS = np.arange(
    -60, 60, 3
)  # range(-90, 91, 3) # at MPI-B IMOD only works for ranges
DETECTOR_SNR = [1.0, 2.0]  # 0.2 # [.15, .25]
MALIGN_MN = 1
MALIGN_MX = 1.5
MALIGN_SG = 0.2

# OUTPUT FILES
OUT_DIR = os.path.realpath(
    ROOT_PATH + "/data_generated/development_all_features"
)  # '/out_all_tomos_9-10' # '/only_actin' # '/out_rotations'
os.makedirs(OUT_DIR, exist_ok=True)

TEM_DIR = OUT_DIR + "/tem"
TOMOS_DIR = OUT_DIR + "/tomos"
os.makedirs(TOMOS_DIR, exist_ok=True)
os.makedirs(TEM_DIR, exist_ok=True)

# OUTPUT LABELS
LBL_MB = 1
LBL_AC = 2
LBL_MT = 3
LBL_CP = 4
LBL_MP = 5
# LBL_BR = 6

##### Main procedure

# set_stomos = SetTomos()
vx_um3 = (VOI_VSIZE * 1e-4) ** 3

# Preparing intermediate directories
lio.clean_dir(TEM_DIR)
lio.clean_dir(TOMOS_DIR)

# Loop for tomograms
for tomod_id in range(NTOMOS):

    print("GENERATING TOMOGRAM NUMBER:", tomod_id)
    hold_time = time.time()

    # Generate the VOI and tomogram density
    if isinstance(VOI_SHAPE, str):
        voi = lio.load_mrc(VOI_SHAPE) > 0
        voi_off = np.zeros(shape=voi.shape, dtype=bool)
        voi_off[
            VOI_OFFS[0][0] : VOI_OFFS[0][1],
            VOI_OFFS[1][0] : VOI_OFFS[1][1],
            VOI_OFFS[2][0] : VOI_OFFS[2][1],
        ] = True
        voi = np.logical_and(voi, voi_off)
        del voi_off
    else:
        voi = np.zeros(shape=VOI_SHAPE, dtype=bool)
        voi[
            VOI_OFFS[0][0] : VOI_OFFS[0][1],
            VOI_OFFS[1][0] : VOI_OFFS[1][1],
            VOI_OFFS[2][0] : VOI_OFFS[2][1],
        ] = True
        voi_inital_invert = np.invert(voi)
    bg_voi = voi.copy()
    voi_voxels = voi.sum()
    tomo_lbls = np.zeros(shape=VOI_SHAPE, dtype=np.float32)
    tomo_den = np.zeros(shape=voi.shape, dtype=np.float32)
    synth_tomo = SynthTomo()
    poly_vtp, mbs_vtp, skel_vtp = None, None, None
    entity_id = 1
    mb_voxels, ac_voxels, mt_voxels, cp_voxels, mp_voxels = 0, 0, 0, 0, 0
    set_mbs = None

    # Membranes loop
    count_mbs, hold_den = 0, None
    for p_id, p_file in enumerate(MEMBRANES_LIST):

        print("\tPROCESSING FILE:", p_file)

        # Loading the membrane file
        memb = MbFile()
        memb.load_mb_file(ROOT_PATH + "/" + p_file)

        # Generating the occupancy
        hold_occ = memb.get_occ()
        if hasattr(hold_occ, "__len__"):
            # hold_occ random number generation between the two values
            hold_occ = np.random.uniform(hold_occ[0], hold_occ[1])
            #hold_occ = TODO: OccGen(hold_occ).gen_occupancy()

        # Membrane random generation by type
        hold_max_rad = memb.get_max_rad()
        if hold_max_rad is None:
            hold_max_rad = math.sqrt(3) * max(VOI_SHAPE) * VOI_VSIZE
        param_rg = (
            memb.get_min_rad(),
            hold_max_rad,
            memb.get_max_ecc(),
        )
        if memb.get_type() == "sphere":
            #mb_sph_generator = SphGen(radius_rg=(param_rg[0], param_rg[1]))
            set_mbs = SetMembranes(
                voi,
                VOI_VSIZE,
                #mb_sph_generator,
                None,
                param_rg,
                memb.get_thick_rg(),
                memb.get_layer_s_rg(),
                hold_occ,
                memb.get_over_tol(),
                bg_voi=bg_voi,
            )
            set_mbs.build_set(verbosity=True)
            hold_den = set_mbs.get_tomo()
            if memb.get_den_cf_rg() is not None:
                hold_den *= random.uniform(
                    memb.get_den_cf_rg()[0], memb.get_den_cf_rg()[1]
                )
        else:
            print("ERROR: Membrane type", memb.get_type(), "not recognized!")
            sys.exit()

        # Density tomogram updating
        voi = set_mbs.get_voi()
        mb_mask = set_mbs.get_tomo() > 0
        mb_mask[voi_inital_invert] = False
        tomo_lbls[mb_mask] = entity_id
        count_mbs += set_mbs.get_num_mbs()
        mb_voxels += (tomo_lbls == entity_id).sum()
        tomo_den = np.maximum(tomo_den, hold_den)
        hold_vtp = set_mbs.get_vtp()
        pp.add_label_to_poly(hold_vtp, entity_id, "Entity", mode="both")
        pp.add_label_to_poly(hold_vtp, LBL_MB, "Type", mode="both")
        if poly_vtp is None:
            poly_vtp = hold_vtp
            skel_vtp = hold_vtp
        else:
            poly_vtp = pp.merge_polys(poly_vtp, hold_vtp)
            skel_vtp = pp.merge_polys(skel_vtp, hold_vtp)
        synth_tomo.add_set_mbs(set_mbs, "Membrane", entity_id, memb.get_type())
    entity_id += 1

    write_mrc_path = TOMOS_DIR + "/tomo_" + str(tomod_id) + "_den.mrc"
    lio.write_mrc(tomo_den, write_mrc_path, v_size=VOI_VSIZE, dtype=np.float32)
