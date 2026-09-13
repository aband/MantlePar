#!/usr/bin/env bash
#cd /home/renpo/Research/MantlePar
#cmake --fresh -S . -B build -DBUILD_TESTING=ON
#cmake --build build --target test_mesh
#ctest --test-dir build -R '^core_mesh_' --output-on-failure

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

cmake --fresh -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DHDF5_ROOT=/home/renpo/system/hdf5-install

# Build all default targets, including every registered test executable.
cmake --build build --parallel

# Run every core test, including mesh_info and math_utils.
ctest --test-dir build \
		  -R '^(core|phase|mfem)_' \
  --output-on-failure --no-tests=error

