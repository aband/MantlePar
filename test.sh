cd /home/renpo/Research/MantlePar
cmake --fresh -S . -B build -DBUILD_TESTING=ON
cmake --build build --target test_mesh
ctest --test-dir build -R '^core_mesh_' --output-on-failure
