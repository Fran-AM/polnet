#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: $0 --config <yaml> --out_dir <dir> [--data_dir <dir>]"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   config_file="$2"; shift 2 ;;
        --out_dir)  out_dir="$2";     shift 2 ;;
        --data_dir) data_dir="$2";    shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "${config_file:-}" ]] && usage
[[ -z "${out_dir:-}" ]]     && usage

mounts=(
    -v "$(realpath "$config_file")":/app/generation_config.yaml:ro
    -v "$(realpath "$out_dir")":/app/outdir
)
[[ -n "${data_dir:-}" ]] && mounts+=(-v "$(realpath "$data_dir")":/app/data:ro)

docker run --rm \
    -e PYTHONUNBUFFERED=1 \
    --user "$(id -u):$(id -g)" \
    "${mounts[@]}" \
    polnet_docker /app/generation_config.yaml