"""This script serves as an alternative entry point for executing all feature extraction"""
from pathlib import Path
import random
import sys
import time

import numpy as np

from polnet import SyntheticSample

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
OUTPUT_DIR = DATA_DIR / "outputs"

# Ensure output directory exists. DATA_DIR should exist already.
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist.")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log("Starting tomogram generation.")

for tomo_id in range(N_TOMOS):
    log(f"Generating tomogram {tomo_id + 1}/{N_TOMOS}")
    hold_time = time.time()

    # Initialize a tomogram
    tomo = SyntheticSample(tomo_id)






