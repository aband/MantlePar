cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DHDF5_ROOT=/home/renpo/system/hdf5-install

cmake --build build --parallel
