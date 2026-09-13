# Eutectic phase visualization

Put `phase_output.h`, `phase_output.cpp`, `export_phase.cpp`, and this
`CMakeLists.txt` in `tools/phase_export/`.

Add the following immediately after `add_subdirectory(src/phase)` in the root
`CMakeLists.txt`:

```cmake
add_subdirectory(tools/phase_export)
```

The phase model remains in `src/phase` without MPI or HDF5 dependencies. This tool
uses PETSc for initialization/options and parallel HDF5 for output, following
the existing mesh exporter. Preserve the root PETSc compiler and MPI launcher
selection. HDF5 must use that same MPI installation.

## Build and run

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DHDF5_ROOT=/home/renpo/system/hdf5-install
cmake --build build --target export_phase --parallel

./build/tools/phase_export/export_phase -output output/phase/diagram

/home/renpo/system/mpich-install/bin/mpiexec -n 4 \
  ./build/tools/phase_export/export_phase \
  -phase_nc 200 -phase_nh 400 -phase_pressure 1e9 \
  -output output/phase/diagram_P1GPa
```

Your updated `build.sh` also builds this target; it disables testing.
With testing enabled, the following runs two serial and two MPI export checks:

```bash
ctest --test-dir build -R '^phase_export_' --output-on-failure --no-tests=error
```

The existing `test.sh` filter `^(core|phase)_` includes these checks automatically.

## Output and ParaView

Each invocation writes one shared `stem.h5` file and a `stem.xdmf` descriptor.
All MPI ranks must see the output directory. Keep the two files together and
open **the `.xdmf` file** in ParaView. The output is an H-C phase diagram, so it
does not depend on the rectangular or perturbed quadrilateral simulation mesh.

The default axes are **x = CD/Xe** and **y = HD - Tep(P)**, with z=0. The plot
covers the component-1-rich range 0 <= CD <= Xe, from just below the eutectic
temperature to above complete melting. `-phase_nc` and `-phase_nh` count cells:
the output has `(nc+1)*(nh+1)` points and `nc*nh` quadrilateral cells.

1. Open the XDMF file and click Apply. Use a front view looking along the z axis.
2. Use Extract Block to select `PhaseDiagram` if you want to display it separately
   from `PhaseBoundaries`. Depending on the reader, these may appear under a
   top-level block named `Phase`.
3. Color the diagram by point field `phil` for melt fraction or `Temperature_K`
   for dimensional temperature. Rescale the color range to the data.
4. For phase regions, choose the **cell field `RegionCell`** and a discrete
   categorical color map. Its integer values are sampled at cell centers, so
   categorical values are not interpolated across the cells.
5. Optionally extract `PhaseBoundaries` as a second display, use a solid color,
   and increase its line width. The lines have a small z offset for visibility.

| Field | Meaning |
|---|---|
| `HD`, `CD` | Actual dimensionless enthalpy and bulk composition inputs |
| `TDp`, `Temperature_K` | Dimensionless and dimensional temperatures |
| `TemperatureAboveEutectic` | TDp - Tep(P) |
| `Enthalpy_Jkg` | HD * cp * (Tm0 - Te0) |
| `phi1`, `phi2`, `phil` | Solid 1, solid 2, and liquid fractions |
| `cl`, `cs` | Liquid and aggregate-solid compositions |
| `HasLiquid`, `HasSolid` | Whether the corresponding composition is meaningful |
| `Region`, `RegionCell` | Phase region at points and at cell centers |
| `dTD_dHD`, `dTD_dCD` | Analytical temperature derivatives |
| `DerivativeFinite` | 0 where the phase model reports a nonfinite derivative |
| `EnthalpyResidual` | TDp + LD*phil - HD |
| `CompositionResidual` | phi2 + phil*cl - CD |
| `FractionResidual` | phi1 + phi2 + phil - 1 |
| `WriterRank` | MPI rank owning each output point; changes with rank count |

All three residuals should be near roundoff. At the singular pure-component
melting corner, `dTD_dCD` is written as a **zero placeholder** and
`DerivativeFinite=0`. Threshold on `DerivativeFinite=1` before interpreting a
derivative plot. Similarly, `cl` and `cs` are zero placeholders when their phases
are absent. A color plot of a derivative can legitimately jump at interfaces.

Region IDs: 1=pure solid, 2=two solids, 3=eutectic melting, 4=solid 1 + liquid,
5=liquid, 6=pure-component melting. The pure-component states lie on the C=0
edge; use the point field `Region` to inspect them. Boundary IDs are:
1=eutectic temperature, 2=end of eutectic melting, 3=liquidus,
4=pure-component melting segment. The eutectic-temperature line is the C>0
solidus and its limit at C=0; the pure component begins melting at Tmp instead.
Boundary segments are clipped to the requested enthalpy window.

## Adjusting the plot and material

```bash
./build/tools/phase_export/export_phase \
  -phase_tm0 2100 -phase_te0 1500 -phase_l 6e5 -phase_cp 1350 \
  -phase_xe 0.3 -phase_pressure 1e9 \
  -phase_nc 300 -phase_nh 500 -output output/phase/modified_material
```

All material inputs have command-line overrides, using SI units:
`-phase_tm0`, `-phase_te0`, `-phase_nu`, `-phase_l`, `-phase_cp`, `-phase_xe`,
`-phase_rho`, `-phase_rhor`, `-phase_mus`, `-phase_mul`, `-phase_k0`, `-phase_g`,
and `-phase_alpha0`. Lowercase `-phase_l` is the latent heat L. Values not supplied
use the model's defaults. The model validates the complete parameter set and
recomputes its derived scales before sampling. Material inputs and derived
scales are stored as named datasets under `/Material` and `/Scales` in HDF5.

Optional plot bounds are `-phase_c_min`, `-phase_c_max`, `-phase_h_min`, and
`-phase_h_max`. C bounds are actual bulk compositions; H bounds are actual HD
values, even when normalized display axes are enabled. Bounds omitted from the
command line are recomputed from the current material and pressure.

Use `-phase_normalized_axes false` for x=CD and y=HD, and
`-phase_boundaries false` to omit the curve block. Since both melting temperatures
have the same pressure shift, subtracting Tep(P) hides that translation on the
normalized axes; use raw axes to compare the absolute enthalpy thresholds at
different pressures.

An application can call `WritePhaseXdmf(comm, model, plot, stem)` again after
`model.setParameters(...)`. Supply identical inputs on all ranks and use a new
stem for each snapshot. The writer does not modify the material or solution.

The collective file/hyperslab pattern follows the
[HDF5 parallel-I/O model](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_par_h_d_f5.html).
The descriptor uses [XDMF structured-grid and attribute definitions](https://www.xdmf.org/index.php/XDMF_Model_and_Format).

## Validation performed

The exporter and driver compiled as C++17 with warnings treated as errors using
test-only PETSc/MPI adapters and a real serial HDF5 library. Generated files were
read with VTK's XDMF reader; dataset shapes, physical fields, conservation
residuals, material overrides, derivative masks, XML filename escaping, and
clipped boundary curves were checked. Simulated row partitions, including empty
ranks, reproduced the serial physical fields exactly. These adapters were used
only for local verification and are not part of this package. Native PETSc/MPI,
parallel HDF5, and CMake/CTest execution still need verification on your machine;
the four registered export checks are provided for that purpose.
