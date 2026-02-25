#!/usr/bin/env bash
set -euo pipefail
# Run from project root
docker build -f docker/Dockerfile -t polnet_docker .