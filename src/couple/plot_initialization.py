#!/usr/bin/env python3
"""Plot cell-center phase quantities, Gauss-point velocities and solver results."""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
from pathlib import Path
import tempfile

# Keep matplotlib's cache writable on compute nodes and in restricted sessions.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mantlepar-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

REGIONS = ["Pure solid", "Two solids", "Eutectic", "Solid + melt", "Liquid", "Pure melting"]
PHASE_FIELDS = [
    ("phi", "Porosity / liquid fraction", "fraction"),
    ("temperature_K", "Temperature", "K"),
    ("phase_pressure_Pa", "Phase pressure", "MPa"),
    ("phi1", "Solid 1 fraction", "fraction"),
    ("phi2", "Solid 2 fraction", "fraction"),
    ("region", "Phase region", ""),
]
STATE_FIELDS = [
    ("h_J_kg", "Specific enthalpy", "J/kg"),
    ("C", "Bulk composition C", "fraction"),
    ("cl", "Liquid composition", "fraction; liquid present"),
    ("cs", "Solid composition", "fraction; solid present"),
    ("dT_dh_K_kg_J", "Temperature derivative dT/dh", "K / (J/kg)"),
    ("dT_dC_K", "Temperature derivative dT/dC", "K per unit composition"),
]

VELOCITY_FIELDS = [
    ("vs_x_m_s", "Solid velocity · x", "cm/year"),
    ("vs_y_m_s", "Solid velocity · y", "cm/year"),
    ("vs_speed", "Solid speed and direction", "cm/year"),
    ("q_x_m_s", "Darcy segregation flux · x", "cm/year"),
    ("q_y_m_s", "Darcy segregation flux · y", "cm/year"),
    ("q_speed", "Darcy flux magnitude and direction", "cm/year"),
]


def load_data(directory):
    """Use the completed manifest, never glob stale files from previous MPI runs."""
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "visualization.json").read_text())
    if metadata.get("schema_version") != 2 or metadata.get("status") != "complete":
        raise ValueError("Visualization output is incomplete or has an unsupported schema. Rerun initialization.")
    if metadata.get("flow_status") != "solved" or "solve" not in metadata:
        raise ValueError("This output predates the initial flow solve. Rerun initialization.")
    if len(metadata["files"]) != metadata["ranks"]:
        raise ValueError("Manifest does not contain every MPI rank.")
    result = {}
    for kind, count_key in [("cells", "cell_rows"), ("edges", "edge_rows"), ("mesh", "mesh_rows"),
                            ("centers", "center_rows"), ("flow_cells", "flow_cell_rows"), ("flow_edges", "flow_edge_rows")]:
        parts = []
        for entry in metadata["files"]:
            path = (directory / entry[kind]).resolve()
            if path.parent != directory:
                raise ValueError("Manifest files must be in the output directory.")
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                names = reader.fieldnames
                rows = list(reader)
            if len(rows) != entry[count_key]:
                raise ValueError(f"Incomplete or stale data: {path.name}")
            data = np.empty(len(rows), dtype=[(name, "f8") for name in names])
            for name in names:
                data[name] = [float(row[name]) for row in rows]
            parts.append(data)
        result[kind] = np.concatenate(parts)
    m = metadata
    expected = {"cells": m["cell_count"] * m["cell_points_per_axis"]**2,
                "edges": m["edge_count"] * m["edge_points"], "mesh": m["cell_count"], "centers": m["cell_count"],
                "flow_cells": m["cell_count"]*m["cell_points_per_axis"]**2, "flow_edges": 4*m["cell_count"]*m["edge_points"]}
    for kind, data in result.items():
        if len(data) != expected[kind]:
            raise ValueError(f"Missing or duplicate {kind} samples.")
        keys = data[["cell_id"]] if kind == "mesh" else data[["entity_id", "side", "q"]] if kind == "flow_edges" else data[["entity_id", "q"]]
        if len(np.unique(keys)) != len(data):
            raise ValueError(f"Duplicate {kind} sample identifiers.")
        required = data.dtype.names if kind == "mesh" else [n for n in data.dtype.names if n not in ("cl", "cs", "dT_dH", "dT_dC", "dT_dh_K_kg_J", "dT_dC_K")]
        if any(not np.all(np.isfinite(data[n])) for n in required):
            raise ValueError(f"Nonfinite required values in {kind} data.")
    return metadata, result


def values(data, field):
    if field in ("vs_speed", "q_speed"):
        prefix=field.split("_")[0]
        return np.hypot(data[prefix+"_x_m_s"], data[prefix+"_y_m_s"])*100*365*24*3600
    value = data[field].copy()
    if field.endswith("_m_s"):
        value *= 100*365*24*3600
    elif field == "phase_pressure_Pa":
        value /= 1e6
    elif field == "cl":
        value[data["has_liquid"] == 0] = np.nan
    elif field == "cs":
        value[data["has_solid"] == 0] = np.nan
    elif field in ("dT_dH", "dT_dC", "dT_dh_K_kg_J", "dT_dC_K"):
        value[data["derivatives_finite"] == 0] = np.nan
    return value


def plot_fields(metadata, data, mesh, fields, title, path, ranges, cell_center=False):
    factor = metadata["scales"]["length_m"] / 1000
    unit = "km"
    if cell_center:
        mesh = np.sort(mesh, order="cell_id")
        data = np.sort(data, order="entity_id")
        if not np.array_equal(mesh["cell_id"], data["entity_id"]):
            raise ValueError("Cell-center values do not match the mesh.")
    vertices = np.stack([np.column_stack((mesh[f"x{i}"], mesh[f"y{i}"])) for i in range(4)], axis=1) * factor
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 10.0), layout="constrained")
    subtitle=f"Initial time = {metadata['time']*metadata['scales']['time_s']/(365*24*3600):g} years"
    if metadata["flow_porosity"].startswith("prescribed zero"):
        subtitle += " · flow porosity fixed at zero"
    fig.suptitle(f"{metadata['simulation']} · {title}\n{subtitle}", fontsize=15)
    for ax, (field, label, color_unit) in zip(axes.flat, fields):
        ax.add_collection(PolyCollection(vertices, facecolors="none", edgecolors="#495365", linewidths=0.35, alpha=0.3))
        v = values(data, field)
        finite = np.isfinite(v)
        marker_size = max(1.0, min(18.0, 16000 / len(data)))
        if np.any(~finite) and not cell_center:
            ax.scatter(data["x"][~finite]*factor, data["y"][~finite]*factor, s=marker_size, color="#d9dde2", linewidths=0)
        zero_velocity=field in {name for name,_,_ in VELOCITY_FIELDS} and np.all(v==0)
        if zero_velocity:
            ax.scatter(data["x"]*factor,data["y"]*factor,s=marker_size,color="#a9bfd0",linewidths=0)
            ax.text(.5,.5,"0 cm/year throughout",ha="center",va="center",transform=ax.transAxes,
                    bbox=dict(facecolor="white",alpha=.9,edgecolor="none"),fontsize=10)
        elif np.any(finite):
            kwargs = dict(cmap="RdBu_r" if field.endswith(("_x_m_s","_y_m_s")) else "viridis")
            if field == "region":
                kwargs = dict(cmap=ListedColormap(["#435269", "#76a5ae", "#e5bc47", "#d97942", "#b74356", "#9372b2"]),
                              norm=BoundaryNorm(np.arange(0.5, 7), 6))
            elif field in ranges:
                kwargs.update(vmin=ranges[field][0], vmax=ranges[field][1])
            if cell_center:
                points = PolyCollection(vertices, array=np.ma.masked_invalid(v), edgecolors="#495365",
                                        linewidths=0.25, cmap=kwargs["cmap"], norm=kwargs.get("norm"))
                if "vmin" in kwargs:
                    points.set_clim(kwargs["vmin"], kwargs["vmax"])
                points.cmap = points.cmap.copy()
                points.cmap.set_bad("#d9dde2")
                ax.add_collection(points)
                ax.scatter(data["x"]*factor, data["y"]*factor, s=2, color="#263645", alpha=0.35, linewidths=0)
            else:
                points = ax.scatter(data["x"][finite]*factor, data["y"][finite]*factor, c=v[finite],
                                    s=marker_size, linewidths=0, rasterized=True, **kwargs)
            if field in ("vs_speed", "q_speed") and np.max(v)>1e-15:
                # Subsample arrows only; all Gauss points remain in the color map.
                selected = np.linspace(0,len(data)-1,min(120,len(data)),dtype=int)
                prefix=field.split("_")[0]
                span=max(metadata["domain"][1]-metadata["domain"][0], metadata["domain"][3]-metadata["domain"][2])*factor
                scale=np.max(v)/(0.06*span)
                arrows=ax.quiver(data["x"][selected]*factor,data["y"][selected]*factor,
                    values(data,prefix+"_x_m_s")[selected],values(data,prefix+"_y_m_s")[selected],angles="xy",scale_units="xy",scale=scale,
                    color="#142638",alpha=.75,width=.003)
                ax.quiverkey(arrows,.62,.94,np.max(v),f"{np.max(v):.2g} cm/year",labelpos="E",coordinates="axes",fontproperties={"size":8})
            bar = fig.colorbar(points, ax=ax, shrink=0.78, pad=0.03)
            bar.ax.tick_params(labelsize=8)
            if field == "region":
                bar.set_ticks(range(1, 7), labels=REGIONS)
            else:
                bar.set_label(color_unit, fontsize=9)
                ax.text(0.02, 0.02, f"min {v[finite].min():.4g}\nmax {v[finite].max():.4g}",
                        transform=ax.transAxes, fontsize=8, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        else:
            if cell_center:
                ax.add_collection(PolyCollection(vertices,facecolors="#d9dde2",edgecolors="#495365",linewidths=.25))
            ax.text(0.5, 0.5, "Not defined\nat these points", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        domain = np.array(metadata["domain"]) * factor
        ax.set(xlim=domain[:2], ylim=domain[2:], xlabel=f"x ({unit})", ylabel=f"y ({unit})")
        ax.set_title(label,fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def settings_tables(m):
    s = m["solver"]
    solver_rows = []
    for name, block in [("Coupled system", s), ("Velocity block", s["velocity"]), ("Pressure Schur block", s["pressure"])]:
        solver_rows.append([name, block["method"], block["preconditioner"], f"{block['rtol']:.3g}",
                            f"{block['atol']:.3g}", f"{block['dtol']:.3g}", str(block["max_iterations"])])
    yes = lambda key: "Yes" if s[key] else "No"
    policy_rows = [
        ["True residual required", yes("require_true_residual"), "Check ||b−Ax|| ≤ max(atol, rtol·||b||)."],
        ["Inner solves must converge", yes("require_subsolver_convergence"), "Reject a solve if a monitored block solver failed."],
        ["Solver error on nonconvergence", yes("error_if_not_converged"), "Generic solver policy; initialization always requires an accepted solve."],
        ["Nonzero initial guess", yes("initial_guess_nonzero"), "Configured starting-guess policy for initialization."],
        ["Pressure nullspace", s["pressure_nullspace"], "Requested pressure mode, validated against the assembled operator."],
        ["Project right-hand side", yes("project_rhs"), "Allow projection onto the compatible pressure subspace."],
        ["Remove pressure gauge", yes("remove_pressure_nullspace"), "Remove the validated nullspace component after solving."],
        ["PETSc option prefix", s["options_prefix"], "PETSc runtime options override configured defaults."],
    ]
    t = m["transport"]
    transport_rows = [["Lax–Friedrichs stabilization", t["lf_mode"]], ["Flux linearization", t["lf_linearization"]],
                      ["Thermal diffusivity (m²/s)", f"{t['thermal_diffusivity']*m['scales']['length_m']**2/m['scales']['time_s']:.5g}"],
                      ["Normal diffusion samples", str(t["diffusion_samples"])],
                      ["Sampling extent fraction", f"{t['diffusion_extent']:.5g}"]]
    if t["lf_mode"] == "global":
        transport_rows.append(["Supplied global wave-speed bound (cm/year)", f"{t['global_speed']*m['scales']['velocity_m_s']*100*365*24*3600:.5g}"])
    r = m["solve"]
    result_rows = [["Accepted", "Yes" if r["converged"] else "No"],
                   ["Effective outer method / preconditioner", f"{r['method']} / {r['preconditioner']}"],
                   ["Outer iterations", str(r["iterations"])], ["PETSc convergence reason", str(r["reason"])],
                   ["True residual ||b−Ax||", f"{r['true_residual']:.5e}"],
                   ["Relative true residual", f"{r['relative_true_residual']:.5e}"],
                   ["Effective true residual threshold", f"{r['true_residual_threshold']:.5e}"],
                   ["All monitored subsolvers converged", "Yes" if r["subsolvers_converged"] else "No"],
                   ["Pressure gauge removed", "Yes" if r["pressure_gauge_removed"] else "No"],
                   ["Removed RHS component norm", f"{r['removed_rhs_component']:.5e}"]]
    inner_rows = [[b["prefix"], str(b["solves"]), str(b["iterations"]), str(b["failures"])] for b in r["subsolvers"]]
    return [(["Measured initial flow solve", "Value"], result_rows),
            (["Subsolver", "Calls", "Total iterations", "Failed calls"], inner_rows),
            (["Level", "Method", "Preconditioner", "Relative tol.", "Absolute tol.", "Divergence tol.", "Max. iterations"], solver_rows),
            (["Acceptance / gauge setting", "Value", "Meaning"], policy_rows),
            (["Transport setting", "Value"], transport_rows)]


def report(metadata, data, images, destination):
    m = metadata
    tables = settings_tables(m)
    summary = (f"{m['mesh_family']}, {m['nx']} × {m['ny']} cells, {m['ranks']} MPI rank(s). "
               f"{len(data['centers']):,} cell centers; {len(data['flow_cells']):,} cell Gauss points and {len(data['flow_edges']):,} cell-side edge samples.")
    notes = [
        "Phase panels evaluate the prescribed H/C profiles and the equilibrium phase model at mapped cell centers (the mean of four vertices). Each cell is colored by that center value; these are point evaluations, not averages of Gauss samples.",
        f"Pressure is the {m['pressure_model']} pressure supplied to the phase model, shown in MPa. Solved Stokes and rescaled Darcy pressures are included in the flow Gauss-point CSV files and flow_dofs_rank_*.csv.",
        "Gray cells indicate absent-phase compositions or undefined temperature derivatives. Liquid and solid compositions are meaningful only where those phases are present.",
        "Velocity colors show all Gauss points in cm/year, with a subset of direction arrows. Solid velocity is u0·us. Darcy segregation flux is u0·phi^(1+theta)·ud, using the flow porosity at each point (harmonic on edges); it is a volume flux per unit area, not liquid velocity. Edge panels retain both cell traces. Cell averages remain in initial_rank_*.csv.",
        "The initial coupled Darcy–Stokes solve is complete. Measured convergence results appear below, followed by configured solver defaults. PETSc runtime overrides may change those defaults.",
        "Relative tolerance scales with the right-hand-side norm; absolute tolerance sets a residual floor. Divergence tolerance limits residual growth. The Schur preset uses S = E − D A⁻¹ G with full block factorization.",
    ]
    notes.insert(3, f"Flow coefficient: {m['flow_porosity']}. Equilibrium porosity in the phase panels remains a thermodynamic diagnostic; the dry-preheat example explicitly supplies zero to flow assembly.")
    checks = [[name.title(), f"{value:.3e}"] for name, value in m["maximum_balance_errors"].items()]
    tables.append((["Maximum absolute phase-balance residual", "Value"], checks))
    def html_table(headers, rows):
        return "<div class='table'><table><thead><tr>" + "".join(f"<th>{html.escape(v)}</th>" for v in headers) + \
            "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table></div>"
    sections = []
    for title, filename in images:
        encoded = base64.b64encode((destination/filename).read_bytes()).decode("ascii")
        sections.append(f"<section><h2>{html.escape(title)}</h2><img alt='{html.escape(title)}' src='data:image/png;base64,{encoded}'></section>")
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Initialization report</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#243348;background:#f3f5f8;margin:0}main{max-width:1160px;margin:auto;padding:32px}
h1{font-size:32px;margin-bottom:8px}h2{font-size:23px}header,section{background:white;border:1px solid #dfe5ed;border-radius:12px;padding:24px;margin-bottom:20px}
.label{color:#49617f;font-size:13px;text-transform:uppercase;letter-spacing:.08em}.note{border-left:4px solid #558397;padding-left:16px}img{width:100%;height:auto}
.table{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #e2e7ed}th{background:#eef2f6}footer{font-size:13px;color:#52647a}
</style><main>"""
    document += f"<header><div class='label'>MantlePar · Initial state</div><h1>{html.escape(m['simulation'])}</h1><p>{html.escape(summary)}</p>"
    document += "<div class='note'>" + "".join(f"<p>{html.escape(note)}</p>" for note in notes[:4]) + "</div></header>"
    document += "".join(sections)
    document += "<section><h2>Solver settings</h2>" + "".join(f"<p>{html.escape(note)}</p>" for note in notes[5:])
    document += "".join(html_table(*table) for table in tables) + "</section>"
    document += f"<footer><p>{html.escape(notes[4])}</p><p>Input: {html.escape(m['source'])}</p></footer></main></html>"
    (destination / "index.html").write_text(document, encoding="utf-8")
    lines = [f"# {m['simulation']} — initialization", "", summary, ""] + [note+"\n" for note in notes]
    for headers, rows in tables:
        lines += ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        lines += ["| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |" for row in rows]
        lines.append("")
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing visualization.json")
    parser.add_argument("--output-dir", type=Path, help="Default: <input-dir>/visualization")
    args = parser.parse_args()
    metadata, data = load_data(args.input_dir)
    destination = args.output_dir or args.input_dir / "visualization"
    destination.mkdir(parents=True, exist_ok=True)
    images = []
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    for group, fields in [("phase", PHASE_FIELDS), ("state", STATE_FIELDS)]:
        filename=f"centers_{group}.png"
        title="Cell centers · " + ("phase and pressure" if group=="phase" else "state and derivatives")
        plot_fields(metadata,data["centers"],data["mesh"],fields,title,destination/filename,{},cell_center=True)
        images.append((title,filename))
    ranges={}
    for field,_,_ in VELOCITY_FIELDS:
        v=np.concatenate([values(data[kind],field) for kind in ("flow_cells","flow_edges")])
        lo,hi=float(v.min()),float(v.max())
        if field.endswith(("_x_m_s","_y_m_s")):
            bound=max(abs(lo),abs(hi))
            if bound>0: ranges[field]=(-bound,bound)
        elif hi>lo: ranges[field]=(lo,hi)
    for kind,label in [("flow_cells","Cell Gauss points"),("flow_edges","Edge Gauss points · both cell traces")]:
        filename=f"{kind}_velocity.png"
        title=label+" · velocity"
        plot_fields(metadata,data[kind],data["mesh"],VELOCITY_FIELDS,title,destination/filename,ranges)
        images.append((title,filename))
    report(metadata, data, images, destination)
    print(f"Initialization report: {(destination / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
