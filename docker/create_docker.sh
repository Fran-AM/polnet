#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --imod <path/to/imod_installer.sh>"
    echo ""
    echo "  --imod   Path to a locally downloaded IMOD Linux .sh installer."
    echo "           Download from: https://bio3d.colorado.edu/imod/download.html"
    exit 1
}

imod_path=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --imod) imod_path="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "${imod_path:-}" ]] && usage
[[ ! -f "$imod_path" ]] && { echo "Error: IMOD installer not found: $imod_path"; exit 1; }

# Stage installer inside the Docker build context, then clean up on exit
installer_staged="docker/imod_installer.sh"
cp "$imod_path" "$installer_staged"
trap "rm -f $installer_staged" EXIT

# Run from project root
docker build -f docker/Dockerfile -t polnet_docker .