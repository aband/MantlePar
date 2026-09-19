#!/usr/bin/env bash
# Evolve an existing completed preheat, then generate individual and combined videos.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$PROJECT_ROOT/src/couple/run_evolution.py" "$@"
