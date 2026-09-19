#!/usr/bin/env python3
"""Native end-to-end checks of quadrature output, phase masks and MPI ownership."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_initialization import load_data, values


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def run(args, text, name, ranks=1):
    root = (args.output_dir / name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    text = text.replace("directory: output/{mesh_family}", "directory: " + json.dumps(str(root)))
    # Split x so PETSc's global ordering differs from x-fast natural ordering.
    text = text.replace("ranks: 1", f"ranks: {ranks}").replace("process_grid: [1, 1]", f"process_grid: [{ranks}, 1]")
    path = root / "case.yaml"
    path.write_text(text)
    command = [str(args.executable)]
    if ranks > 1:
        command = [args.mpiexec, args.numproc_flag, str(ranks)] + [s for s in args.mpi_preflags.split(";") if s] + command
        command += [s for s in args.mpi_postflags.split(";") if s]
    subprocess.run(command + ["-input", str(path)], check=True, timeout=90)
    return root, *load_data(root)


def check_quadrature(root, m, data):
    handoff = json.loads((root / "transport_state.json").read_text())
    check(handoff["schema_version"] == 1 and handoff["status"] == "complete", "Incomplete H/C export")
    check(handoff["representation"] == "cell averages" and handoff["ordering"] == "j*nx+i", "Incorrect H/C layout")
    check(handoff["H"]["units"] == "h/(cp*dT)" and handoff["C"]["units"] == "fraction", "Incorrect legacy units")
    for key in ("nx", "ny"):
        check(handoff["mesh"][key] == m[key], "Incorrect H/C mesh dimensions")
    check(handoff["mesh"]["family"] == m["mesh_family"], "Incorrect H/C mesh family")
    np.testing.assert_array_equal(handoff["mesh"]["domain"], m["domain"])
    for key, scale in handoff["scales"].items():
        check(scale == m["scales"][key], "Incorrect H/C reference scales")
    check(handoff["time"] == m["time"], "Incorrect H/C export time")
    np.testing.assert_allclose(handoff["time_s"], m["time"]*m["scales"]["time_s"])
    # Parse the same plain whitespace values as legacy fscanf, with no header.
    restart = {key: np.loadtxt(root/handoff[key]["file"], ndmin=2) for key in ("H", "C")}
    for field in restart.values():
        check(field.shape == (m["ny"], m["nx"]) and np.all(np.isfinite(field)), "Malformed H/C matrix")
    cell = data["cells"]
    np.testing.assert_allclose(cell["weight"].sum(), (m["domain"][1]-m["domain"][0])*(m["domain"][3]-m["domain"][2]), rtol=2e-12)
    for kind in ("cells", "edges", "centers"):
        d = data[kind]
        check(np.all(d["weight"] > 0), "Nonpositive physical quadrature weights")
        np.testing.assert_allclose(d["phi1"]+d["phi2"]+d["phi"], 1, atol=2e-12)
        np.testing.assert_allclose(d["T"]+m["scales"]["latent_heat"]*d["phi"], d["H"], atol=2e-12)
        check(np.all(np.isnan(d["cl"][d["has_liquid"] == 0])), "Absent liquid composition was not masked")
        check(np.all(np.isnan(d["cs"][d["has_solid"] == 0])), "Absent solid composition was not masked")
    # Integrals of Gauss values must recover the independently stored Vec data.
    for entry in m["files"]:
        averages = np.atleast_1d(np.genfromtxt(root / f"initial_rank_{entry['rank']:06d}.csv", delimiter=",", names=True))
        for column, source, scale in [("h_J_kg", "H", m["scales"]["enthalpy_Jkg"]),
                                      ("temperature_K", "T", m["scales"]["temperature_K"]),
                                      ("x_centroid_m", "x_centroid", m["scales"]["length_m"]),
                                      ("y_centroid_m", "y_centroid", m["scales"]["length_m"]),
                                      ("area_m2", "area", m["scales"]["length_m"]**2)]:
            np.testing.assert_allclose(averages[column], averages[source]*scale, rtol=2e-13)
        for row in averages:
            for field in restart:
                check(restart[field][int(row["j"]), int(row["i"])] == row[field], "Export changed or reordered stored H/C averages")
            points = cell[cell["entity_id"] == row["cell_id"]]
            np.testing.assert_allclose(points["weight"].sum(), row["area"], rtol=2e-12)
            for field in ("H", "C", "T", "phi"):
                mean = np.dot(points["weight"], points[field]) / row["area"]
                np.testing.assert_allclose(mean, row[field], rtol=2e-12, atol=2e-13)
    if m["pressure_model"] == "lithostatic":
        for kind in ("cells", "edges", "centers"):
            np.testing.assert_allclose(data[kind]["phase_pressure_Pa"], -3000*10*m["scales"]["length_m"]*data[kind]["y"], rtol=2e-12)
    centers=np.sort(data["centers"],order="entity_id")
    if m["mesh_family"] == "perturbed_quadrilateral":
        check(np.max(np.abs(restart["C"].ravel()-centers["C"])) > 1e-6,
              "H/C handoff incorrectly used center samples instead of nonlinear-profile averages")
    mesh=np.sort(data["mesh"],order="cell_id")
    for axis in ("x","y"):
        expected=sum(mesh[f"{axis}{k}"] for k in range(4))/4
        np.testing.assert_allclose(centers[axis],expected,atol=2e-15)
    for kind in ("cells","edges","centers"):
        d=data[kind]
        np.testing.assert_allclose(d["h_J_kg"],d["H"]*m["scales"]["enthalpy_Jkg"],rtol=2e-13)
        np.testing.assert_allclose(d["x_m"],d["x"]*m["scales"]["length_m"],rtol=2e-13)
        np.testing.assert_allclose(d["dT_dh_K_kg_J"],d["dT_dH"]*m["scales"]["temperature_K"]/m["scales"]["enthalpy_Jkg"],equal_nan=True)
        np.testing.assert_allclose(d["dT_dC_K"],d["dT_dC"]*m["scales"]["temperature_K"],equal_nan=True)
    for kind in ("flow_cells","flow_edges"):
        d=data[kind]
        for axis in ("x","y"):
            np.testing.assert_allclose(d[f"vs_{axis}_m_s"],d[f"us_{axis}"]*m["scales"]["velocity_m_s"],rtol=2e-13)
            np.testing.assert_allclose(d[f"q_{axis}_m_s"],d[f"ud_{axis}"]*d["flow_phi"]**(1+m["flow_theta"])*m["scales"]["velocity_m_s"],rtol=2e-13)
        np.testing.assert_allclose(values(d,"vs_speed"),np.hypot(d["vs_x_m_s"],d["vs_y_m_s"])*100*365*24*3600,rtol=2e-13)
    # Check FE continuity and MPI ghost reconstruction on shared edges.
    edges=np.sort(data["flow_edges"],order=["edge_id","q","entity_id"])
    same=(edges["edge_id"][1:]==edges["edge_id"][:-1]) & (edges["q"][1:]==edges["q"][:-1])
    a,b=edges[:-1][same],edges[1:][same]
    for field in ("x","y","us_x","us_y","flow_phi"):
        np.testing.assert_allclose(a[field],b[field],rtol=2e-10,atol=2e-12)
    for row_a,row_b in zip(a,b):
        polygon=mesh[int(row_a["entity_id"])]; side=int(row_a["side"])
        dx=polygon[f"x{(side+1)%4}"]-polygon[f"x{side}"]
        dy=polygon[f"y{(side+1)%4}"]-polygon[f"y{side}"]
        jump=(row_a["ud_x"]-row_b["ud_x"])*dy-(row_a["ud_y"]-row_b["ud_y"])*dx
        check(abs(jump)<2e-11,"Discontinuous Darcy normal velocity")
    check(m["flow_status"] == "solved" and m["solve"]["converged"], "Initial flow was not solved")
    check(m["solve"]["true_residual"] <= m["solve"]["true_residual_threshold"], "Flow residual exceeds tolerance")
    check(m["solve"]["subsolvers_converged"], "An inner solve failed")
    check(m["solver"]["rtol"] == 1e-10 and m["solver"]["require_true_residual"], "Solver settings lost")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mpiexec")
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--mpi-preflags", default="")
    parser.add_argument("--mpi-postflags", default="")
    parser.add_argument("--phase-limits", action="store_true")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    args.executable = args.executable.resolve()
    text = args.input.read_text()
    root, m, data = run(args, text, "serial")
    check_quadrature(root, m, data)
    area = (m["domain"][1]-m["domain"][0])*(m["domain"][3]-m["domain"][2])
    preheat="formalpreheat" in m["simulation"]
    np.testing.assert_allclose(np.dot(data["cells"]["weight"], data["cells"]["H"]), (3.35 if preheat else 3.5)*area, rtol=2e-12)
    if preheat:
        check(m["flow_porosity"]=="prescribed zero (legacy dry preheat)","Legacy dry flow assumption lost")
        check(np.max(data["centers"]["phi"])>0,"Equilibrium phase diagnostics were replaced by dry flow coefficients")
        np.testing.assert_allclose(data["centers"]["C"],.01-.12*data["centers"]["y"],atol=2e-14)
        for kind in ("flow_cells","flow_edges"):
            d=data[kind]
            check(np.all(d["flow_phi"]==0) and np.all(d["q_x_m_s"]==0) and np.all(d["q_y_m_s"]==0),"Dry flow has nonzero porosity or segregation flux")
        edge=data["flow_edges"]
        bottom=edge[np.abs(edge["y"]+.5)<1e-13]
        np.testing.assert_allclose(values(bottom,"vs_y_m_s"),3.2,atol=2e-9)
        top=edge[np.abs(edge["y"])<1e-13]
        vertices=np.linspace(0,.5,m["nx"]+1)
        boundary=np.array([3.2*math.erf(x)/math.erf(.1) if x<.1 else 3.2 for x in vertices])
        np.testing.assert_allclose(values(top,"vs_x_m_s"),np.interp(top["x"],vertices,boundary),atol=2e-8)
        np.testing.assert_allclose(values(top,"vs_y_m_s"),0,atol=2e-9)
    exact_c = 0.04*area
    if m["mesh_family"] == "perturbed_quadrilateral":
        exact_c = 0.03*area + 0.02*(0.08*math.sqrt(2*math.pi)*math.erf(0.2/(0.08*math.sqrt(2))))*(0.1*math.sqrt(2*math.pi)*math.erf(0.2/(0.1*math.sqrt(2))))
    np.testing.assert_allclose(np.dot(data["cells"]["weight"], data["cells"]["C"]), exact_c, rtol=2e-10)
    # Stale files must not be mixed into a new decomposition.
    (root / "cell_gauss_rank_999999.csv").write_text("stale,invalid\n")
    load_data(root)
    if args.mpiexec:
        mpi_root, mpi_m, mpi_data = run(args, text, "mpi", ranks=2)
        check_quadrature(mpi_root, mpi_m, mpi_data)
        for filename in ("cellH1.dat", "cellC1.dat"):
            np.testing.assert_allclose(np.loadtxt(root/filename), np.loadtxt(mpi_root/filename), rtol=2e-13, atol=2e-14)
        def dofs(directory, metadata):
            rows = np.concatenate([np.atleast_1d(np.genfromtxt(directory/e["flow_dofs"], delimiter=",", names=True,
                                   dtype=None, encoding="utf-8")) for e in metadata["files"]])
            check(len(np.unique(rows[["field", "natural_dof"]])) == len(rows), "Duplicate flow DOFs")
            return np.sort(rows, order=["field", "natural_dof"])
        serial_dofs, mpi_dofs = dofs(root,m), dofs(mpi_root,mpi_m)
        np.testing.assert_array_equal(serial_dofs["field"], mpi_dofs["field"])
        np.testing.assert_array_equal(serial_dofs["natural_dof"], mpi_dofs["natural_dof"])
        np.testing.assert_allclose(serial_dofs["value"], mpi_dofs["value"], atol=2e-7, rtol=2e-7)
        for kind in data:
            keys = ["cell_id"] if kind == "mesh" else ["entity_id", "side", "q"] if kind=="flow_edges" else ["entity_id", "q"]
            serial = np.sort(data[kind], order=keys)
            parallel = np.sort(mpi_data[kind], order=keys)
            for field in serial.dtype.names:
                np.testing.assert_allclose(serial[field], parallel[field], rtol=2e-7 if kind.startswith("flow_") else 2e-12, atol=2e-9 if kind.startswith("flow_") else 2e-13, equal_nan=True)
    if args.phase_limits:
        initial = "name: affine\n      parameters: {value: 3.0, origin: [0.0, 0.0], gradient: [0.0, -2.5]}"
        for name, h in [("pure_melting_corner", 2000/520), ("fully_solid", 2.0)]:
            limits = text.replace(initial, f"name: constant\n      parameters: {{value: {h:.17g}}}")
            limits = limits.replace("parameters: {value: 0.04}", "parameters: {value: 0.0}")
            limits = limits.replace("model: lithostatic\n    surface_y: 0.0", "model: constant\n    value_pa: 0.0")
            # The dry Stokes Schur action can lose CG definiteness near zero
            # when its inner velocity action terminates at an absolute tolerance.
            # GMRES handles this limit without changing residual acceptance.
            limits = limits.replace("    pressure:\n      ksp: cg", "    pressure:\n      ksp: gmres")
            limit_root, limit_m, limit_data = run(args, limits, name)
            check_quadrature(limit_root, limit_m, limit_data)
            if name == "pure_melting_corner":
                check(np.all(limit_data["cells"]["derivatives_finite"] == 0), "Singular derivative was not flagged")
                check(np.all(np.isnan(values(limit_data["cells"], "dT_dC"))), "Singular derivative was plotted")
            else:
                check(np.all(limit_data["cells"]["has_liquid"] == 0), "Solid state has liquid composition")
        if args.plots:
            # Plot the absent-liquid case to exercise entirely masked panels.
            subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "plot_initialization.py"),
                            "--input-dir", str(limit_root)], check=True, timeout=90)
            check((limit_root/"visualization/index.html").stat().st_size > 10000, "Missing self-contained report")
            check(all((limit_root/"visualization"/name).is_file() for name in ("centers_phase.png","centers_state.png","flow_cells_velocity.png","flow_edges_velocity.png")), "Missing figure panels")
    # A failed/partial write must not be presented as completed output.
    manifest = root / "visualization.json"
    original = manifest.read_text()
    manifest.write_text('{"schema_version":2,"status":"incomplete"}')
    try:
        try:
            load_data(root)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted incomplete output")
    finally:
        manifest.write_text(original)
    print("Passed: physical quadrature, initialized averages, phase masks, solver metadata and file ownership.")


if __name__ == "__main__":
    main()
