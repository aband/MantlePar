#!/usr/bin/env bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DMANTLE_BUILD_DRIVER=ON \
  -DHDF5_ROOT=/home/renpo/system/hdf5-install

cmake --build build --parallel
