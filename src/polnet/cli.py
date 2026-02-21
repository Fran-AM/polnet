"""Command line interface for polnet."""
import argparse
import logging
from pathlib import Path

import yaml

from .logging_conf import setup_logger, _LOGGER as logger
from .main import gen_tomos


def app():
    """Main function for the command line interface."""
    parser = argparse.ArgumentParser(
        description="Polnet: A tool for generating synthetic tomograms "
                    "with complex membrane structures.",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity. Use -v for INFO, -vv for DEBUG.",
    )
    parser.add_argument(
        "-o", "--log-dir",
        type=str,
        default=None,
        help="Directory for log files (default: output folder from config).",
    )
    args = parser.parse_args()

    # --- Load config first (we need output folder for the log) ---
    config_path = Path(args.config)
    if not config_path.exists():
        # Can't use logger yet — print directly
        logger.error(f"Error: Configuration file {config_path} does not exist.")
        return 1

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- Determine log directory ---
    if args.log_dir is not None:
        log_dir = Path(args.log_dir)
    else:
        root = config["folders"]["root"]
        if root is None:
            root = Path(__file__).parents[2]
        log_dir = Path(root) / config["folders"]["output"]

    # --- Map verbosity count to log level ---
    if args.verbose >= 2:
        console_level = logging.DEBUG
    elif args.verbose == 1:
        console_level = logging.INFO
    else:
        console_level = logging.WARNING

    # --- Initialize logging (BEFORE any work) ---
    setup_logger(log_folder=log_dir, console_level=console_level)

    logger.debug(f"Config loaded from {config_path}")
    logger.debug(f"Console log level: {logging.getLevelName(console_level)}")

    gen_tomos(config)


if __name__ == "__main__":
    app()