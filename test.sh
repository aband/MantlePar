#cd /home/renpo/Research/MantlePar
#cmake --fresh -S . -B build -DBUILD_TESTING=ON
#cmake --build build --target test_mesh
#ctest --test-dir build -R '^core_mesh_' --output-on-failure

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

cmake --fresh -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DHDF5_ROOT=/home/renpo/system/hdf5-install

cmake --build build --target test_mesh test_integral -j

ctest --test-dir build \
  -R '^core_(mesh|integral)_' \
  --output-on-failure --no-tests=error
