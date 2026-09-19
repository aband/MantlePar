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
    if fields == VELOCITY_FIELDS:
        from movie_fields import VelocityDrawing, reference_speed, velocity_vectors
        gauss = np.column_stack([data[name] for name in ('vs_x_m_s','vs_y_m_s','q_x_m_s','q_y_m_s','flow_phi')])
        points = np.column_stack((data['x'],data['y']))*factor
        refs = [reference_speed(v) for v in velocity_vectors(gauss)]
        fig, axes = plt.subplots(3,2,figsize=(12.8,14.4))
        fig.subplots_adjust(left=.075,right=.94,bottom=.07,top=.9,hspace=.35,wspace=.3)
        drawing = VelocityDrawing(fig,axes,points,np.asarray(metadata['domain'])*factor,refs)
        drawing.draw(gauss)
        fig.suptitle(f"{metadata['simulation']} · {title}",fontsize=15)
        fig.text(.5,.03,'Gauss-point arrows · Constant-color streamlines · No magnitude heatmap or mesh overlay',ha='center',fontsize=11)
        fig.savefig(path,dpi=140); plt.close(fig)
        return
    if cell_center:
        mesh = np.sort(mesh, order="cell_id")
        data = np.sort(data, order="entity_id")
        if not np.array_equal(mesh["cell_id"], data["entity_id"]):
            raise ValueError("Cell-center values do not match the mesh.")
    vertices = np.stack([np.column_stack((mesh[f"x{i}"], mesh[f"y{i}"])) for i in range(4)], axis=1) * factor
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 10.0), layout="constrained")
    subtitle=f"State time = {metadata['time']*metadata['scales']['time_s']/(365*24*3600):g} years"
    if metadata["flow_porosity"].startswith("prescribed zero"):
        subtitle += " · flow porosity fixed at zero"
    fig.suptitle(f"{metadata['simulation']} · {title}\n{subtitle}", fontsize=15)
    for ax, (field, label, color_unit) in zip(axes.flat, fields):
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
                points = PolyCollection(vertices, array=np.ma.masked_invalid(v), edgecolors="none",
                                        linewidths=0, cmap=kwargs["cmap"], norm=kwargs.get("norm"))
                if "vmin" in kwargs:
                    points.set_clim(kwargs["vmin"], kwargs["vmax"])
                points.cmap = points.cmap.copy()
                points.cmap.set_bad("#d9dde2")
                ax.add_collection(points)
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
    levels=[("Coupled system", s)]
    if s["preconditioner"]=="full Schur field split":
        levels += [("Velocity block", s["velocity"]), ("Pressure Schur block", s["pressure"])]
    for name, block in levels:
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
    if not inner_rows: inner_rows=[["No block subsolvers used", "—", "—", "—"]]
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
        "Velocity panels show quivers at a subset of the actual Gauss points and constant-color streamlines, without magnitude heatmaps or mesh overlays. Arrow references are in cm/year. Solid velocity is vs=u0·us; Darcy segregation flux is q=u0·phi^(1+theta)·ud; liquid velocity is vl=vs+q/phi where phi>0. Liquid velocity is undefined in dry regions. Edge quivers retain sample traces; coincident traces are averaged only for streamline interpolation. Streamlines are instantaneous interpolated flow lines, not particle paths.",
        "The initial coupled Darcy–Stokes solve is complete. Measured convergence results appear below, followed by configured solver defaults. PETSc runtime overrides may change those defaults.",
        "Relative tolerance scales with the right-hand-side norm; absolute tolerance sets a residual floor. Divergence tolerance limits residual growth. The Schur preset uses S = E − D A⁻¹ G with full block factorization.",
    ]
    notes.insert(3, f"Flow coefficient: {m['flow_porosity']}. Equilibrium porosity in the phase panels remains a thermodynamic diagnostic; the dry-preheat example explicitly supplies zero to flow assembly.")
    coupled=m.get("evolution_model")=="full phase coupling"
    evolved=m.get("accepted_steps",0)>0 or coupled
    if evolved:
        notes[0]="Phase panels evaluate the evolved H/C cell-average reconstructions at mapped cell centers. The original initialization profiles are not reused. Unique-edge phase CSV values use the canonical left-cell trace, or the interior trace at boundaries."
        notes[3]=(f"Legacy dry preheat completed {m['accepted_steps']} forward Euler steps. H evolves by solid-flow advection and diffusion of H; C and the initial dry flow remain fixed. "
                  "The dry thermal approximation uses H·dT as temperature for diffusion. Equilibrium temperature and porosity shown here are separate diagnostics and do not feed back into this preheat solve.")
        sampled=data["cells"]["h_J_kg"]
        if sampled.min()<0:
            notes[3]+=(f" The unlimited ML-WENO cell Gauss samples span {sampled.min():.5g} to {sampled.max():.5g} J/kg, "
                       "including negative reconstruction overshoots near the cooled boundary. Gauss-point phase diagnostics inherit these artifacts; no positivity limiter or clipping is applied. The exported H/C matrices contain the cell averages.")
        notes[5]="The initial Darcy–Stokes solve supplies the fixed velocity throughout preheat. The measured convergence results below describe that initial flow solve. Evolved H/C are exported in cellH1.dat and cellC1.dat; preheat_history.csv records accepted steps and the heat balance."
        tables.insert(0,(["Preheat integration", "Value"], [
            ["Accepted forward Euler steps", str(m["accepted_steps"])],
            ["Final time (years)", f"{m['time']*m['scales']['time_s']/(365*24*3600):.7g}"],
            ["Composition / flow", "Held fixed"],
            ["Diffused quantity", "H (legacy dry thermal approximation)"],
        ]))
    if coupled:
        notes[3]=(f"Full phase coupling completed {m['accepted_steps']} SSPRK2 steps. Both H and C evolve; equilibrium phase and Darcy–Stokes flow are recomputed at each stage and at the final time. "
                  "Enthalpy includes sensible and latent heat transport, conduction of phase temperature, and adiabatic cooling. Composition uses the phase-weighted effective velocity; chemical diffusion is neglected. "
                  "The mean-preserving bounded ML-WENO reconstruction leaves imported cell averages unchanged. Edge porosity uses the harmonic mean of both phase traces.")
        notes[5]="Measured convergence describes the final coupled flow solve. Exact imported averages are in starting_cellH.dat / starting_cellC.dat; final averages are in cellH1.dat / cellC1.dat. evolution_history.csv records both conservation balances, accepted steps, melt fraction, and flow residuals. The evolution clock starts at zero after preheat; source_state.json records the preheat time."
        if tables[0][0][0]=="Preheat integration": tables.pop(0)
        tables.insert(0,(["Coupled integration", "Value"], [
            ["Accepted SSPRK2 steps",str(m["accepted_steps"])],
            ["Elapsed evolution time (years)",f"{m['time']*m['scales']['time_s']/(365*24*3600):.7g}"],
            ["Flow evaluation time (years)",f"{m['flow_time']*m['scales']['time_s']/(365*24*3600):.7g}"],
            ["Diffused quantity","Equilibrium temperature"],
            ["Reference","reviewed.pdf, equations 2.81 and 4.11–4.14; SSPRK2 3.210"],
        ]))
        for k,(headers,rows) in enumerate(tables):
            if headers[0]=="Measured initial flow solve": tables[k]=(["Measured final flow solve","Value"],rows)
        if m["solver"]["preconditioner"].startswith("sparse LU"):
            notes[6]="Sparse LU preconditions the outer FGMRES solve. A small shift is applied only to the factors to handle zero pressure pivots; the outer solve and acceptance residual use the unchanged flow operator. Relative tolerance scales with the RHS norm; absolute tolerance sets a residual floor."
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
<title>MantlePar state report</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#243348;background:#f3f5f8;margin:0}main{max-width:1160px;margin:auto;padding:32px}
h1{font-size:32px;margin-bottom:8px}h2{font-size:23px}header,section{background:white;border:1px solid #dfe5ed;border-radius:12px;padding:24px;margin-bottom:20px}
.label{color:#49617f;font-size:13px;text-transform:uppercase;letter-spacing:.08em}.note{border-left:4px solid #558397;padding-left:16px}img{width:100%;height:auto}
.table{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #e2e7ed}th{background:#eef2f6}footer{font-size:13px;color:#52647a}
</style><main>"""
    document += f"<header><div class='label'>MantlePar · {'Phase-coupled evolution' if coupled else 'Evolved preheat state' if evolved else 'Initial state'}</div><h1>{html.escape(m['simulation'])}</h1><p>{html.escape(summary)}</p>"
    document += "<div class='note'>" + "".join(f"<p>{html.escape(note)}</p>" for note in notes[:4]) + "</div></header>"
    document += "".join(sections)
    document += "<section><h2>Solver settings</h2>" + "".join(f"<p>{html.escape(note)}</p>" for note in notes[5:])
    document += "".join(html_table(*table) for table in tables) + "</section>"
    document += f"<footer><p>{html.escape(notes[4])}</p><p>Input: {html.escape(m['source'])}</p></footer></main></html>"
    (destination / "index.html").write_text(document, encoding="utf-8")
    lines = [f"# {m['simulation']} — {'phase-coupled evolution' if coupled else 'preheat' if evolved else 'initialization'}", "", summary, ""] + [note+"\n" for note in notes]
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
    if metadata.get("accepted_steps",0)>0 and metadata.get("evolution_model")!="full phase coupling":
        history=np.atleast_1d(np.genfromtxt(args.input_dir/"preheat_history.csv",delimiter=",",names=True))
        if len(history)!=metadata["accepted_steps"]+1 or not np.isclose(history["time"][-1],metadata["time"]):
            raise ValueError("Preheat history does not match the exported final state.")
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
        t=history["time_years"]
        axes[0].fill_between(t,history["h_min_J_kg"],history["h_max_J_kg"],alpha=.2,color="#267c93",label="Cell-average range")
        axes[0].plot(t,history["h_mean_J_kg"],color="#267c93",label="Domain mean")
        axes[0].set(xlabel="Time (years)",ylabel="Specific enthalpy (J/kg)",title="Preheat enthalpy history")
        axes[0].legend()
        axes[1].plot(t[1:],history["dt_s"][1:]/(365*24*3600),color="#87509b")
        axes[1].set(xlabel="Time (years)",ylabel="Step duration (years)",title="Accepted time steps")
        for ax in axes: ax.grid(alpha=.2)
        filename="preheat_history.png"; fig.savefig(destination/filename,dpi=160); plt.close(fig)
        images.insert(0,("Preheat time evolution",filename))
    if metadata.get("evolution_model")=="full phase coupling":
        history=np.atleast_1d(np.genfromtxt(args.input_dir/"evolution_history.csv",delimiter=",",names=True))
        if len(history)!=metadata["accepted_steps"]+1 or not np.isclose(history["time"][-1],metadata["time"]):
            raise ValueError("Evolution history does not match the final state.")
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
        t=history["time_years"]
        for ax,mean,low,high,label in [
            (axes[0,0],"h_mean_J_kg","h_min_J_kg","h_max_J_kg","Specific enthalpy (J/kg)"),
            (axes[0,1],"C_mean","C_min","C_max","Bulk composition (fraction)")]:
            ax.fill_between(t,history[low],history[high],alpha=.2,label="Cell-average range")
            ax.plot(t,history[mean],label="Domain mean"); ax.set(ylabel=label); ax.legend()
        axes[1,0].plot(t,history["phi_max"],label="Maximum cell-average melt fraction")
        axes[1,0].set(ylabel="Porosity (fraction)"); axes[1,0].legend()
        axes[1,1].plot(t[1:],history["dt"][1:]*metadata["scales"]["time_s"]/(365*24*3600))
        axes[1,1].set(ylabel="Accepted step (years)")
        for ax in axes.flat: ax.set(xlabel="Elapsed evolution time (years)"); ax.grid(alpha=.2)
        filename="evolution_history.png"; fig.savefig(destination/filename,dpi=160); plt.close(fig)
        images.insert(0,("Coupled evolution of energy, composition and melt",filename))
        mesh=np.sort(data["mesh"],order="cell_id")
        factor=metadata["scales"]["length_m"]/1000
        vertices=np.stack([np.column_stack((mesh[f"x{i}"],mesh[f"y{i}"])) for i in range(4)],axis=1)*factor
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
        for ax,field,scale,label in [(axes[0],"H",metadata["scales"]["enthalpy_Jkg"],"Enthalpy change (J/kg)"),
                                      (axes[1],"C",1,"Composition change (fraction)")]:
            before=np.loadtxt(args.input_dir/f"starting_cell{field}.dat")
            after=np.loadtxt(args.input_dir/f"cell{field}1.dat")
            if before.shape!=(metadata["ny"],metadata["nx"]) or after.shape!=before.shape:
                raise ValueError("H/C change matrices do not match the mesh.")
            change=(after-before).ravel()*scale
            bound=max(float(np.abs(change).max()),1e-30)
            colored=PolyCollection(vertices,array=change,cmap="RdBu_r",edgecolors="none",clim=(-bound,bound))
            ax.add_collection(colored); ax.autoscale_view(); ax.set_aspect("equal")
            ax.set(xlabel="x (km)",ylabel="y (km)",title=label)
            fig.colorbar(colored,ax=ax,label=label)
        filename="transport_change.png"; fig.savefig(destination/filename,dpi=160); plt.close(fig)
        images.insert(1,("Change from the imported preheat cell averages",filename))
    report(metadata, data, images, destination)
    print(f"State report: {(destination / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
