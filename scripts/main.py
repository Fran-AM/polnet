"""This script serves as an alternative entry point for executing all feature extraction"""
from pathlib import Path
import random
import time

import numpy as np

from polnet import SynthTomo, logger, setup_logger
from polnet.utils import lio

# Custom print function for consistent logging
def log(message):
    # log as: [HH:MM:SS] message
    print(f"[{time.strftime('%H:%M:%S', time.gmtime(time.time()))}] {message}")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Generation parameters
N_TOMOS = 2
VOI_SHAPE = (
    500,
    500,
    250,
)
VOI_OFFS = (
    4,
    4,
    4,
)
VOI_VSIZE = 10  # A/vx

MEMBRANES_LIST = [
    "in_mbs/sphere.mbs",
    "in_mbs/ellipse.mbs",
    # "in_mbs/toroid.mbs",
]

HELIX_LIST = [
    # "in_helix/mt.hns",
    # "in_helix/actin.hns"
]

PROTEINS_LIST = [
    "in_10A/4v4r_10A.pns",
    "in_10A/3j9i_10A.pns",
    "in_10A/4v4r_50S_10A.pns",
    "in_10A/4v4r_30S_10A.pns",
    "in_10A/6utj_10A.pns",
    "in_10A/5mrc_10A.pns",
    "in_10A/4v7r_10A.pns",
    "in_10A/2uv8_10A.pns",
    "in_10A/4v94_10A.pns",
    "in_10A/4cr2_10A.pns",
    # "in_10A/3qm1_10A.pns",
    # "in_10A/3h84_10A.pns",
    # "in_10A/3gl1_10A.pns",
    # "in_10A/3d2f_10A.pns",
    # "in_10A/3cf3_10A.pns",
    # "in_10A/2cg9_10A.pns",
    # "in_10A/1u6g_10A.pns",
    # "in_10A/1s3x_10A.pns",
    # "in_10A/1qvr_10A.pns",
    # "in_10A/1bxn_10A.pns",
]

MB_PROTEINS_LIST = [
    # "in_10A/mb_6rd4_10A.pms",
    # "in_10A/mb_5wek_10A.pms",
    # "in_10A/mb_4pe5_10A.pms",
    # "in_10A/mb_5ide_10A.pms",
    # "in_10A/mb_5gjv_10A.pms",
    # "in_10A/mb_5kxi_10A.pms",
    # "in_10A/mb_5tj6_10A.pms",
    # "in_10A/mb_5tqq_10A.pms",
    # "in_10A/mb_5vai_10A.pms",
]

# Directory paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = DATA_DIR / "data_generated" / "output"

# Ensure output directory exists. DATA_DIR should exist already.
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist.")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize logger
setup_logger(log_folder=OUTPUT_DIR)
logger_header = "Starting tomogram generation script\n"
logger_header += f" - Output directory: {OUTPUT_DIR}\n"
logger_header += f" - Number of tomograms to generate: {N_TOMOS}\n"
logger_header += f" - VOI shape (vx): {VOI_SHAPE}\n"
logger_header += f" - VOI voxel size (A/vx): {VOI_VSIZE}\n"
logger_header += f" - VOI offset (vx): {VOI_OFFS}\n"
logger_header += f" - Membranes list: {MEMBRANES_LIST}\n"
logger_header += f" - Helices list: {HELIX_LIST}\n"
logger_header += f" - Proteins list: {PROTEINS_LIST}\n"
logger_header += f" - Membrane proteins list: {MB_PROTEINS_LIST}\n"
logger.info(logger_header)
del logger_header

# Generate tomograms
for tomo_id in range(N_TOMOS):
    logger.info("-" * 50)
    logger.info(f"Generating tomogram {tomo_id + 1}/{N_TOMOS}")
    hold_time = time.time()

    synth_tomo = SynthTomo(
        id=tomo_id + 1,
        mbs_file_list=MEMBRANES_LIST,
        hns_file_list=HELIX_LIST,
        pns_file_list=PROTEINS_LIST,
        pms_file_list=MB_PROTEINS_LIST,
    )
    
    logger.info("Starting sample generation")
    synth_tomo.gen_sample(
        data_path=DATA_DIR,
        shape=VOI_SHAPE,
        v_size=VOI_VSIZE,
        offset=VOI_OFFS,
        verbosity=False
    )
    logger.info(f"Sample generation completed in {time.time() - hold_time:.2f} seconds.")
    hold_time = time.time()
    logger.debug("Starting tilt series simulation. TODO: TEM")
    # TODO: TEM
    logger.debug(f"TEM simulation finished in {time.time() - hold_time:.2f} seconds. TODO: TEM ")
    hold_time = time.time()
    logger.info("Saving tomogram to disk.")
    synth_tomo.save_tomo(output_folder=OUTPUT_DIR / f"Tomo{tomo_id + 1:03d}")
    logger.info(f"Tomogram saved in folder {OUTPUT_DIR / f'Tomo{tomo_id + 1:03d}'} in {time.time() - hold_time:.2f} seconds.")
    synth_tomo.summary()










