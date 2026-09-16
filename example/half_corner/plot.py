#!/usr/bin/env python3
"""Plot cell fields from a successful half-corner run, using its actual mesh."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from run import read_config


def read_results(path):
    config = read_config(path)
    if config.get("studies", {}).get("convergence", {}).get("enabled", False):
        raise ValueError("This plotter expects one steady solve per input; disable the convergence study.")
    family = config["mesh"]["family"]
    tokens = dict(mesh_family=family, level=0, step=0)
    directory = path.parent / config["output"]["directory"].format(**tokens)
    with (directory / "solver_report.csv").open(newline="") as stream:
        reports = list(csv.DictReader(stream))
    if len(reports) != 1:
        raise ValueError(f"Expected one solver report in {directory}")
    report = reports[0]
    if report["converged"] != "1" or report["subsolvers_converged"] != "1":
        raise ValueError(f"Solver did not converge: {directory / 'solver_report.csv'}")
    if report["family"] != family or int(report["level"]) != 0:
        raise ValueError("Solver report does not match the selected mesh family/level.")
    if read_config(directory / "input_used.yaml") != config:
        raise ValueError("Input changed since this run; rerun before plotting these fields.")
    try:
        import h5py
    except ImportError as error:
        raise ValueError("Install plotting dependencies: python3 -m pip install -r example/common/requirements.txt") from error
    filename = directory / config["output"]["fields"]["file"].format(**tokens)
    names = ("coordinates", "topology", "cell_centers", "porosity",
             "stokes_velocity", "darcy_segregation_flux", "stokes_pressure_assembled")
    with h5py.File(filename, "r") as stream:
        data = {name: np.asarray(stream[name]) for name in names}
    nx, ny = config["mesh"]["cells"]
    count = nx * ny
    topology = data["topology"]
    if topology.shape != (count, 4) or not np.issubdtype(topology.dtype, np.integer):
        raise ValueError("Expected one quadrilateral per mesh cell.")
    coordinates = data["coordinates"]
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Expected XYZ mesh coordinates.")
    if topology.min() < 0 or topology.max() >= len(coordinates):
        raise ValueError("Topology contains an invalid vertex index.")
    for name in ("cell_centers", "stokes_velocity", "darcy_segregation_flux"):
        if data[name].shape != (count, 3):
            raise ValueError(f"Unexpected shape for {name}")
    for name in ("porosity", "stokes_pressure_assembled"):
        if data[name].shape != (count, 1):
            raise ValueError(f"Unexpected shape for {name}")
    if any(not np.isfinite(values).all() for values in data.values()):
        raise ValueError("Nonfinite values in the plotted fields or mesh.")
    if np.any(data["porosity"] < 0) or np.any(data["porosity"] >= 1):
        raise ValueError("Cell porosity is outside [0, 1).")
    return config, directory, report, data


def plot_fields(config, directory, report, data):
    """Render polygons without interpolating across the wet/dry interface."""
    polygons = data["coordinates"][data["topology"], :2]
    centers = data["cell_centers"][:, :2]
    solid = data["stokes_velocity"][:, :2]
    flux = data["darcy_segregation_flux"][:, :2]
    nx, ny = config["mesh"]["cells"]
    phi = data["porosity"].ravel()
    panels = (
        ("Cell-average porosity", phi, None, "fraction"),
        ("Solid velocity", np.linalg.norm(solid, axis=1), solid, r"$|u_s|$"),
        ("Segregation flux", np.linalg.norm(flux, axis=1), flux, r"$|q|$"),
        ("Assembled Stokes pressure", data["stokes_pressure_assembled"].ravel(), None, r"$p_s$"),
    )
    xlim, ylim = config["mesh"]["domain"]["x"], config["mesh"]["domain"]["y"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
    stride = max(1, int(np.ceil(max(nx, ny) / 20)))
    indices = np.arange(nx * ny).reshape(ny, nx)[::stride, ::stride].ravel()
    for ax, (title, values, vectors, label) in zip(axes.flat, panels):
        collection = PolyCollection(polygons, array=values, cmap="viridis", edgecolors="none", rasterized=True)
        if title == "Cell-average porosity":
            maximum = config["porosity"]["prescribed_function"]["parameters"]["maximum_porosity"]
            collection.set_clim(0, max(float(maximum), np.finfo(float).eps))
        ax.add_collection(collection)
        fig.colorbar(collection, ax=ax, label=label, shrink=0.85, pad=0.025)
        if vectors is not None:
            selected = indices[np.linalg.norm(vectors[indices], axis=1) > 0]
            if len(selected):
                chosen = vectors[selected]
                maximum = float(np.linalg.norm(chosen, axis=1).max())
                spacing = min((xlim[1] - xlim[0]) / nx, (ylim[1] - ylim[0]) / ny) * stride
                arrows = ax.quiver(centers[selected, 0], centers[selected, 1], chosen[:, 0], chosen[:, 1],
                                   angles="xy", scale_units="xy", scale=maximum / (1.4 * spacing),
                                   width=0.003, color="black", pivot="mid")
                ax.quiverkey(arrows, 0.06, 1.03, maximum, f"{maximum:.2e}", labelpos="E", coordinates="axes")
        ax.set(xlim=xlim, ylim=ylim, xlabel="x", ylabel="y")
        ax.set_title(title, pad=30 if vectors is not None else 12)
        ax.set_aspect("equal")
    family = config["mesh"]["family"].replace("_", " ")
    residual = float(report["relative_true_residual"])
    fig.suptitle(f"Half-corner flow — {family} — {nx} × {ny}\n"
                 f"Dimensionless fields; true relative residual {residual:.2e}", fontsize=13)
    formats = config["output"].get("plots", {}).get("formats", ["png"])
    try:
        for extension in formats:
            if extension not in ("png", "pdf"):
                raise ValueError(f"Unsupported plot format: {extension}")
            target = directory / f"half_corner.{extension}"
            fig.savefig(target, dpi=180)
            print(target)
    finally:
        plt.close(fig)
    print(f"  cell-average porosity: [{phi.min():.6g}, {phi.max():.6g}]")
    print(f"  max |u_s|={np.linalg.norm(solid, axis=1).max():.6e}; "
          f"max |q|={np.linalg.norm(flux, axis=1).max():.6e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path,
                        default=[Path(__file__).resolve().with_name("input.yaml")])
    args = parser.parse_args()
    for path in args.inputs:
        path = path.expanduser().resolve()
        if path.is_dir():
            path /= "input.yaml"
        plot_fields(*read_results(path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        print(f"Half-corner plot: {error}", file=sys.stderr)
        raise SystemExit(2)
