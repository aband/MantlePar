#!/usr/bin/env bash
# Run dry preheat, then phase-coupled evolution on the same mesh, and make videos.
# Example: ./run.sh --grid 20 --preheat-final-time 15000 --final-time 5
# Warm surface endpoint: ./run.sh --grid 30 --warm-end 0.05
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$PROJECT_ROOT/src/couple/run_evolution.py" --run-preheat "$@"
