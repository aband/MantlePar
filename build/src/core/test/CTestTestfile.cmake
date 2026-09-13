# CMake generated Testfile for 
# Source directory: /home/renpo/Research/MantlePar/src/core/test
# Build directory: /home/renpo/Research/MantlePar/build/src/core/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[core_mesh_serial]=] "/home/renpo/Research/MantlePar/build/src/core/test/test_mesh")
set_tests_properties([=[core_mesh_serial]=] PROPERTIES  TIMEOUT "60" _BACKTRACE_TRIPLES "/home/renpo/Research/MantlePar/src/core/test/CMakeLists.txt;4;add_test;/home/renpo/Research/MantlePar/src/core/test/CMakeLists.txt;0;")
add_test([=[core_mesh_mpi_2x1]=] "/usr/bin/mpiexec" "-n" "2" "/home/renpo/Research/MantlePar/build/src/core/test/test_mesh" "-mesh_px" "2" "-mesh_py" "1")
set_tests_properties([=[core_mesh_mpi_2x1]=] PROPERTIES  PROCESSORS "2" TIMEOUT "60" _BACKTRACE_TRIPLES "/home/renpo/Research/MantlePar/src/core/test/CMakeLists.txt;13;add_test;/home/renpo/Research/MantlePar/src/core/test/CMakeLists.txt;0;")
add_test([=[core_mesh_mpi_1x2]=] "/usr/bin/mpiexec" "-n" "2" "/home/renpo/Research/MantlePar/build/src/core/test/test_mesh" "-mesh_px" "1" "-mesh_py" "2")
set_tests_properties([=[core_mesh_mpi_1x2]=] PROPERTIES  PROCESSORS "2" TIMEOUT "60" _BACKTRACE_TRIPLES "/home/renpo/Research/MantlePar/src/core/test/CMakeLists.txt;13;add_test;/home/renpo/Research/MantlePar/src/core/test/CMakeLists.txt;0;")
add_test([=[core_mesh_mpi_2x2]=] "/usr/bin/mpiexec" "-n" "4" "/home/renpo/Research/MantlePar/build/src/core/test/test_mesh" "-mesh_px" "2" "-mesh_py" "2")
set_tests_properties([=[core_mesh_mpi_2x2]=] PROPERTIES  PROCESSORS "4" TIMEOUT "60" _BACKTRACE_TRIPLES "/home/renpo/Research/MantlePar/src/core/test/CMakeLists.txt;13;add_test;/home/renpo/Research/MantlePar/src/core/test/CMakeLists.txt;0;")
