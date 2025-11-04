"""This script serves as an alternative entry point for executing all feature extraction"""
from pathlib import Path
import random
import time

import numpy as np

from polnet import SyntheticSample, MbFile
from polnet.utils import lio

# Custom print function for consistent logging
def log(message):
    # log as: [HH:MM:SS] message
    print(f"[{time.strftime('%H:%M:%S', time.gmtime(time.time()))}] {message}")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Generation parameters
N_TOMOS = 10
VOI_SHAPE = (
    300,
    300,
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
    "in_mbs/toroid.mbs",
]

# Output labels
OUTPUT_LABELS = {
    "membranes": 1
}

# Directory paths
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR = DATA_DIR / "data_generated" / "output"

# Ensure output directory exists. DATA_DIR should exist already.
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist.")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log("Starting tomogram generation.")

for tomo_id in range(N_TOMOS):
    log(f"Generating tomogram {tomo_id + 1}/{N_TOMOS}")
    hold_time = time.time()

    # Initialize a tomogram
    tomo = SyntheticSample(
        id = tomo_id,
        shape = VOI_SHAPE,
        v_size = VOI_VSIZE,
        offset = VOI_OFFS
    )

    # Generating membranes and adding them to the tomogram
    for mb_file_rpath in MEMBRANES_LIST:
        mb_file_apath = DATA_DIR / mb_file_rpath

        mb_file = MbFile()
        mb_params = mb_file.load(mb_file_apath)

        log(f"Generating membranes of type {mb_file.type} from file: {mb_file_apath.name}")

        tomo.add_set_membranes(
            params=mb_params,
            max_mbtries=10,
            verbosity=True
        )

        log (f"Membranes of type {mb_file.type} added to tomogram {tomo_id}.")

    log(f"Tomogram {tomo_id} generation time (s): {time.time() - hold_time:.2f}\n")

    log(f"Saving tomogram {tomo_id} density and labels.")

    write_mrc_path = OUTPUT_DIR / f"tomo_{tomo_id}_den.mrc"
    lio.write_mrc(
        tomo.density,
        write_mrc_path,
        v_size=VOI_VSIZE,
        dtype=np.float32
    )








