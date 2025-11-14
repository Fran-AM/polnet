import argparse
import json
from pathlib import Path

from polnet import gen_tomos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tomograms using PolNet.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration JSON file."
    )
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    funcion = {"gen_tomos": gen_tomos}

    funcion = PolnetFactory.get_function("gen_tomos")

    funcion(config)