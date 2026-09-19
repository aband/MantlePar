#!/usr/bin/env python3
"""End-to-end legacy preheat: flux sign/area, Euler order, CFL, export and MPI."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

from test_visualization import check_quadrature
from plot_initialization import load_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--executable", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mpiexec")
    p.add_argument("--numproc-flag", default="-n")
    args = p.parse_args()
    source = args.input.read_text().replace("directory: output/{mesh_family}", "directory: .")

    def run(name, text, grid=(1, 1), fail=False):
        root = (args.output_dir/name).resolve()
        root.mkdir(parents=True, exist_ok=True)
        text = text.replace("ranks: 1", f"ranks: {grid[0]*grid[1]}")
        text = text.replace("process_grid: [1, 1]", f"process_grid: [{grid[0]}, {grid[1]}]")
        path = root/"input.yaml"
        path.write_text(text)
        command = [str(args.executable.resolve()), "-input", str(path)]
        if grid != (1, 1):
            command = [args.mpiexec, args.numproc_flag, str(grid[0]*grid[1])] + command
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if fail:
            assert result.returncode != 0, "Invalid run was accepted"
            return root, result.stdout+result.stderr
        if result.returncode:
            raise AssertionError(result.stdout+result.stderr)
        m, data = load_data(root)
        check_quadrature(root, m, data)
        h, c = np.loadtxt(root/"cellH1.dat"), np.loadtxt(root/"cellC1.dat")
        # C remains the initial affine profile, with physical cell centroids.
        for entry in m["files"]:
            a = np.atleast_1d(np.genfromtxt(root/f"initial_rank_{entry['rank']:06d}.csv", names=True, delimiter=","))
            np.testing.assert_allclose(a["C"], .01-.12*a["y_centroid"], rtol=2e-12, atol=2e-13)
        history = np.atleast_1d(np.genfromtxt(root/"preheat_history.csv", names=True, delimiter=","))
        assert len(history) == m["accepted_steps"]+1
        assert np.max(np.abs(history["balance_error"])) < 2e-12
        np.testing.assert_allclose(np.diff(history["H_integral"]),
                                   -history["dt"][1:]*history["outward_H_flux"][1:], atol=2e-12)
        np.testing.assert_allclose(history["time"][-1], m["time"], atol=1e-14)
        assert np.all(history["courant"] <= .5+1e-12)
        assert m["evolution_model"] == "legacy dry preheat"
        assert all(np.all(data[k]["flow_phi"] == 0) for k in ("flow_cells", "flow_edges"))
        return root, m, data, h, c, history

    clipped = source.replace("end: 15000.0", "end: 25.0")
    serial = run("serial", clipped)
    assert serial[1]["accepted_steps"] == 3
    np.testing.assert_array_equal(serial[-1]["dt"], [0, 10, 10, 5])
    assert serial[3][-1, -1] < 3.3, "Surface did not cool"
    assert serial[2]["centers"]["H"].min() < 3.3, "Visualization reused initial H profile"
    if args.mpiexec:
        for grid in ((2, 1), (1, 2)):
            parallel = run(f"mpi_{grid[0]}x{grid[1]}", clipped, grid)
            for k in (3, 4):
                np.testing.assert_allclose(serial[k], parallel[k], rtol=3e-10, atol=3e-11)
            np.testing.assert_allclose(serial[-1]["H_integral"], parallel[-1]["H_integral"], atol=3e-12)
        print("Passed: preheat serial/MPI x/y partitions and evolved exports")
        return

    one = run("one_step", source.replace("end: 15000.0", "end: 10.0"))
    expected = np.full((20, 20), 3.35)
    # Independent endpoint formula: sum constant interior samples cancels the
    # endpoint derivative coefficient 25/12. Interior ray length is dx.
    dx = .5/20
    expected[-1, 1:] -= 10*8e-8*(25/12)/(.9*dx/4)*3.35/dx
    np.testing.assert_allclose(one[3], expected, atol=3e-10)
    constant = source.replace("end: 15000.0", "end: 25.0").replace(
        "- {region: top, type: prescribed_temperature, value: {name: constant, parameters: {value: 0.0}}}",
        "- {region: top, type: prescribed_temperature, value: {name: constant, parameters: {value: 3.35}}}")
    conserved = run("constant", constant)
    np.testing.assert_allclose(conserved[3], 3.35, atol=3e-10)
    warped = run("warped", clipped.replace("family: rectangular", "family: perturbed_quadrilateral"))
    assert warped[3].min() < 3.3
    bad = source.replace("end: 15000.0", "end: 1000.0").replace(
        "initial: 10.0, minimum: 10.0, maximum: 10.0", "initial: 1000.0, minimum: 1.0, maximum: 1000.0")
    _, error = run("oversized", bad, fail=True)
    assert "exceeds advection/diffusion limit" in error
    adaptive = run("cfl", bad.replace("control: fixed", "control: cfl").replace("end: 1000.0", "end: 300.0"))
    assert adaptive[1]["time"] == 300
    _, error = run("step_limit", clipped.replace("maximum_steps: 1500", "maximum_steps: 2"), fail=True)
    assert "maximum_steps reached before time.end" in error

    # A 2x2 motionless box gives an exact spatially discrete eigenmode: all
    # four equal cells have two cooled faces, and all interior fluxes vanish.
    # This isolates forward Euler's temporal error from spatial consistency.
    ode = source.replace("cells: [20, 20]", "cells: [2, 2]").replace(
        "start: 15, count: 5", "start: 1, count: 1").replace("interval: [0.0, 0.025]", "interval: [0.0, 0.25]")
    start = ode.index("  boundary_conditions:", ode.index("\ntransport:"))
    end = ode.index("  stabilizer:", start)
    ode = ode[:start]+"""  boundary_conditions:
    enthalpy: {default: {type: zero_flux}}
    composition: {default: {type: zero_flux}}
    temperature:
      default: {type: prescribed_temperature, value: {name: constant, parameters: {value: 2.0}}}
"""+ode[end:]
    start = ode.index("  boundary_conditions:", ode.index("\nflow:"))
    end = ode.index("  linear_system:", start)
    ode = ode[:start]+"""  boundary_conditions:
    stokes:
      components: cartesian
      default:
        x: {type: dirichlet, value: {name: constant, parameters: {value: 0.0}}}
        y: {type: dirichlet, value: {name: constant, parameters: {value: 0.0}}}
    darcy:
      dirichlet_variable: assembled_normal_velocity
      neumann_variable: pressure_potential
      default: {type: dirichlet, value: {name: constant, parameters: {value: 0.0}}}
"""+ode[end:]
    ode = ode.replace("value: [0.0, -1.0]", "value: [0.0, 0.0]").replace("value: 3.35", "value: 2.1")
    ode = ode.replace("pressure_nullspace: none", "pressure_nullspace: provided\n    pressure_modes: {stokes: 1, darcy: 0}")
    ode = ode.replace("remove_pressure_nullspace: false", "remove_pressure_nullspace: true")
    ode = ode.replace("diffusivity: 8.0e-8", "diffusivity: 0.01").replace("end: 15000.0", "end: 0.1")
    decay = 2*.01*(25/12)/(.9*.25/4)/.25
    exact = 2+.1*np.exp(-decay*.1)
    errors = []
    for dt in (.01, .005):
        case = ode.replace("initial: 10.0, minimum: 10.0, maximum: 10.0", f"initial: {dt}, minimum: 1e-12, maximum: {dt}")
        result = run(f"euler_{dt}", case)
        np.testing.assert_allclose(result[3], result[3][0, 0], atol=1e-13)
        errors.append(abs(result[3][0, 0]-exact))
    assert 1.9 < errors[0]/errors[1] < 2.1, errors
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/"plot_initialization.py"),
                    "--input-dir", str(serial[0])], check=True, timeout=90)
    assert (serial[0]/"visualization/preheat_history.png").is_file()
    print("Passed: cooling flux, constant state, heat balance, fixed C, CFL, final clipping, Euler order and evolved plots")


if __name__ == "__main__":
    main()
