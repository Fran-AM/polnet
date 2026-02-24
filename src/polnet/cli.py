"""Command-line interface for polnet.

Entry point registered as ``polnet`` in pyproject.toml.

Usage::

    polnet config/all_features.yaml                     # default (WARNING on console)
    polnet config/all_features.yaml -v                  # INFO on console
    polnet config/all_features.yaml -vv                 # DEBUG on console
    polnet config/all_features.yaml --log-dir /tmp/logs
    polnet config/all_features.yaml -s 12345            # Use random seed
    polnet config/all_features.yaml -n 5                # Generate 5 tomograms
    polnet config/all_features.yaml -v -s 12345 -n 5    # INFO, seed, and multiple tomograms
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .logging_conf import setup_logger, _LOGGER as logger
from .pipeline import gen_tomos

def app() -> int:
    """Parse arguments, initialize logging, and run the tomogram generation pipeline.

    Returns:
        0 on success, 1 on failure.
    """

    parser = config_parser()
    args = parser.parse_args()

    # Load configuration file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file {config_path} does not exist.", file=sys.stderr)
        return 1

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check for first level sections in config before loading
    if "folders" not in config:
        print("Error: Config missing 'folders' section.", file=sys.stderr)
        return 1
    if "global" not in config:
        print("Error: Config missing 'global' section.", file=sys.stderr)
        return 1
    if "sample" not in config:
        print("Error: Config missing 'sample' section.", file=sys.stderr)
        return 1
    if "tem" not in config:
        print("Error: Config missing 'tem' section.", file=sys.stderr)
        return 1

    # Follow up with the logging configuration
    if args.log_dir is not None:
        log_dir = Path(args.log_dir)
    else:
        root = config["folders"].get("root", None)
        if root is None:
            root = Path(__file__).parents[2]
            config["folders"]["root"] = str(root)
        
        if "output" not in config["folders"]:
            print("Error: Config missing 'output' path in 'folders' section.", file=sys.stderr)
            return 1
        
        log_dir = Path(root) / config["folders"]["output"]

    # Check verbosity level and set console log level accordingly
    if args.verbose >= 2:
        console_level = logging.DEBUG
    elif args.verbose == 1:
        console_level = logging.INFO
    else:
        console_level = logging.WARNING

    # Initialize the logger with the determined log directory and console level
    setup_logger(log_folder=log_dir, console_level=console_level)

    logger.debug(f"Config loaded from {config_path}")
    logger.debug(f"Console log level: {logging.getLevelName(console_level)}")

    # Check optional overrides for random seed and number of tomograms
    if args.seed is not None:
        config["global"]["seed"] = args.seed
        logger.debug(f"Random seed overridden to {args.seed}")
    if args.ntomos is not None:
        config["global"]["ntomos"] = args.ntomos
        logger.debug(f"Number of tomograms overridden to {args.ntomos}")

    try:
        gen_tomos(config)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.debug("Traceback:", exc_info=True)
        return 1
    return 0


def config_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for the polnet CLI.
    
    Returns:
        An argparse.ArgumentParser object with the defined arguments.
    """
    parser = argparse.ArgumentParser(
        description="Polnet: a comprehensive tool for generating synthetic cryo-electron tomograms."
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

    parser.add_argument(
        "--version", action="version", version="%(prog)s 1.1.0"
    )

    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Override random seed from config (default: None, use config value).",
    )

    parser.add_argument(
        "-n", "--ntomos",
        type=int,
        default=None,
        help="Number of tomograms to generate, overriding config value (default: None, use config value).",
    )

    return parser

if __name__ == "__main__":
    app()