#!/usr/bin/env python3
"""Plot C++ BRMixed samples; no basis formulas are reimplemented in Python.

Usage: python3 plot_brmixed.py output/brmixed/brmixed.h5
Requires numpy, matplotlib and h5py. Writes PNG/PDF figures next to the HDF5
file in a figures/ directory, or to --out-dir. Does not need MPI or ParaView.
"""

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

try:
    import h5py
except ImportError:
    raise SystemExit("Install the plotting dependencies: python3 -m pip install numpy matplotlib h5py")

CASES = ("rectangular_i", "quadrilateral_i", "rectangular_j", "quadrilateral_j")
MODES = ("shared_start_x", "shared_end_x", "shared_start_y", "shared_end_y",
         "shared_edge", "continuous_velocity")
MODE_LABELS = ("start vertex: x", "end vertex: x", "start vertex: y",
               "end vertex: y", "edge bubble", "combined velocity")
TITLES = {"rectangular_i": "Rectangular · neighbors in i",
          "quadrilateral_i": "Quadrilateral · neighbors in i",
          "rectangular_j": "Rectangular · neighbors in j",
          "quadrilateral_j": "Quadrilateral · neighbors in j"}


def read_data(path):
    data = {}
    with h5py.File(path, "r") as source:
        metadata = np.asarray(source["metadata"])
        if metadata.shape != (6,) or metadata[0] != 1:
            raise ValueError("Unsupported BRMixed output schema")
        subdivisions = int(metadata[1])
        count = (subdivisions + 1)**2
        for name in CASES:
            group = source[name]
            cells = []
            for index in range(2):
                cell = {key: np.asarray(value) for key, value in group[f"cell_{index}"].items()}
                if cell["coordinates"].shape != (count, 3) or cell["connectivity"].shape != (subdivisions**2, 4):
                    raise ValueError(f"Invalid mesh shape in {name}/cell_{index}")
                for field in [f"basis_{k:02d}" for k in range(12)] + list(MODES):
                    if cell[field].shape != (count, 3) or not np.isfinite(cell[field]).all():
                        raise ValueError(f"Invalid field {name}/cell_{index}/{field}")
                cells.append(cell)
            trace = {key: np.asarray(value) for key, value in group["trace"].items()}
            expected = (int(metadata[2]) + 1, len(MODES), 3)
            for field in ("values_a", "values_b", "tangent_a", "tangent_b"):
                if trace[field].shape != expected or not np.isfinite(trace[field]).all():
                    raise ValueError(f"Invalid trace {name}/{field}")
            data[name] = {"cells": cells, "trace": trace,
                          "dofs": np.asarray(group["shared_local_dofs"])}
    return metadata, data


def outline(ax, corners, **kwargs):
    closed = np.concatenate((corners, corners[:1]), axis=0)
    ax.plot(closed[:, 0], closed[:, 1], **kwargs)


def vector_field(ax, cell, name, norm, arrows=7):
    count = int(round(np.sqrt(len(cell["coordinates"]))))
    xyz = cell["coordinates"].reshape(count, count, 3)
    field = cell[name].reshape(count, count, 3)
    magnitude = np.linalg.norm(field[..., :2], axis=-1)
    artist = ax.pcolormesh(xyz[..., 0], xyz[..., 1], magnitude,
                          shading="gouraud", cmap="viridis", norm=norm, rasterized=True)
    indices = np.unique(np.linspace(1, count-2, min(arrows, count-2)).astype(int))
    xx = xyz[..., 0][np.ix_(indices, indices)]
    yy = xyz[..., 1][np.ix_(indices, indices)]
    uu = field[..., 0][np.ix_(indices, indices)]
    vv = field[..., 1][np.ix_(indices, indices)]
    span = max(np.ptp(cell["corners"][:, 0]), np.ptp(cell["corners"][:, 1]))
    ax.quiver(xx, yy, uu, vv, angles="xy", scale_units="xy",
              scale=norm.vmax/(0.12*span), color="white", width=0.006,
              headwidth=3.8, headlength=4.3, pivot="middle")
    outline(ax, cell["corners"], color="#17202a", linewidth=1.25, zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)
    return artist


def save(fig, folder, stem):
    # Replace each destination only after a complete save, so an interrupted
    # rendering does not truncate an existing figure.
    for extension in ("png", "pdf"):
        with tempfile.NamedTemporaryFile(dir=folder, suffix=f".{extension}", delete=False) as output:
            temporary = Path(output.name)
        try:
            fig.savefig(temporary, dpi=180, facecolor="white")
            temporary.replace(folder/f"{stem}.{extension}")
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(fig)


def gallery(data, shape, norm, folder):
    cell = data[f"{shape}_i"]["cells"][0]
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 8.1))
    fig.subplots_adjust(left=0.065, right=0.9, bottom=0.08, top=0.87, wspace=0.3, hspace=0.48)
    sides = ("Bottom", "Right", "Top", "Left")
    for k, ax in enumerate(axes.flat):
        artist = vector_field(ax, cell, f"basis_{k:02d}", norm)
        description = f"vertex {k%4}, {'x' if k < 4 else 'y'}" if k < 8 else f"{sides[k-8]} edge"
        ax.set_title(rf"$\psi_{{{k}}}$ · {description}", fontsize=11, pad=8)
        if k < 4:
            for v, point in enumerate(cell["corners"]):
                ax.annotate(f"v{v}", point[:2], xytext=(3, 5), textcoords="offset points",
                            fontsize=7, bbox={"fc": "white", "ec": "none", "alpha": 0.8}, zorder=6)
        if k >= 8:
            ax.set_xlabel("x", fontsize=10)
        if k % 4 == 0:
            ax.set_ylabel("y", fontsize=10)
    color_ax = fig.add_axes([0.925, 0.18, 0.016, 0.57])
    fig.colorbar(artist, cax=color_ax, label=r"Basis magnitude $|\psi_k|$")
    fig.suptitle(f"BRMixed · {shape} element", x=0.065, ha="left", fontsize=20, weight="bold", y=0.985)
    fig.text(0.065, 0.925, "All 12 local velocity basis functions · physical coordinates · arrows show vector direction",
             fontsize=11, color="#475569")
    save(fig, folder, f"basis_{shape}")


def shared_gallery(data, folder):
    norm = Normalize(0, max(np.linalg.norm(cell[mode][:, :2], axis=1).max()
                            for name in CASES[:2] for cell in data[name]["cells"] for mode in MODES[:5]))
    fig, axes = plt.subplots(2, 5, figsize=(17.2, 5.2))
    fig.subplots_adjust(left=0.055, right=0.92, bottom=0.13, top=0.79, wspace=0.26, hspace=0.65)
    for row, name in enumerate(CASES[:2]):
        patch = data[name]
        for mode, ax in enumerate(axes[row]):
            for index, cell in enumerate(patch["cells"]):
                artist = vector_field(ax, cell, MODES[mode], norm, arrows=5)
                center = cell["corners"][:, :2].mean(axis=0)
                ax.text(*center, "AB"[index], ha="center", va="center", fontsize=8,
                        bbox={"fc": "white", "alpha": 0.85, "ec": "none"}, zorder=6)
            edge = patch["trace"]["coordinates"][[0, -1]]
            ax.plot(edge[:, 0], edge[:, 1], color="white", linewidth=2.2, linestyle="--", zorder=5)
            dof_a, dof_b = patch["dofs"][:, mode]
            ax.set_title(f"{MODE_LABELS[mode]}\nA: {dof_a}  ↔  B: {dof_b}", fontsize=10)
            ax.set_xlabel("x", fontsize=9)
            if mode == 0:
                ax.set_ylabel(f"{name.split('_')[0].capitalize()}\ny", fontsize=10)
    fig.suptitle("Matching basis functions across two neighboring elements", x=0.055, ha="left",
                 fontsize=20, weight="bold", y=0.985)
    fig.text(0.055, 0.92, "Five shared velocity modes · the two cells are sampled independently · dashed line: common edge",
             fontsize=11, color="#475569")
    color_ax = fig.add_axes([0.946, 0.2, 0.012, 0.52])
    fig.colorbar(artist, cax=color_ax, label="Basis magnitude")
    save(fig, folder, "shared_basis")


def metrics(patch):
    trace = patch["trace"]
    a, b = trace["values_a"], trace["values_b"]
    da, db = trace["tangent_a"], trace["tangent_b"]
    length = np.linalg.norm(trace["coordinates"][-1] - trace["coordinates"][0])
    jump = np.max(np.abs(a-b), axis=-1)
    tangent_jump = np.max(np.abs(da-db))
    normal_jump = np.max(np.abs(trace["normals"][0]-trace["normals"][1]))
    normalized = max(np.max(np.abs(a-b)/np.maximum(1, np.maximum(np.abs(a), np.abs(b)))),
                     np.max(length*np.abs(da-db)/np.maximum(1, length*np.maximum(np.abs(da), np.abs(db)))),
                     normal_jump)
    return jump, tangent_jump, normalized


def conformity(data, tolerance, folder):
    fig, axes = plt.subplots(4, 3, figsize=(14.5, 14.4), gridspec_kw={"width_ratios": [1, 1.15, 1.25]})
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.09, top=0.9, wspace=0.36, hspace=0.52)
    colors = plt.get_cmap("tab10")(np.arange(6))
    floor = max(1e-18, tolerance*1e-7)
    for row, name in enumerate(CASES):
        patch = data[name]
        norm = Normalize(0, max(np.linalg.norm(c[MODES[5]][:, :2], axis=1).max() for c in patch["cells"]))
        for index, cell in enumerate(patch["cells"]):
            artist = vector_field(axes[row, 0], cell, MODES[5], norm)
            center = cell["corners"][:, :2].mean(axis=0)
            axes[row, 0].text(*center, "AB"[index], ha="center", fontsize=10,
                              bbox={"fc": "white", "ec": "none", "alpha": 0.85}, zorder=6)
        trace = patch["trace"]
        edge = trace["coordinates"][[0, -1]]
        axes[row, 0].plot(edge[:, 0], edge[:, 1], color="white", linestyle="--", linewidth=2, zorder=5)
        axes[row, 0].set_title(TITLES[name], fontsize=11, weight="bold")
        axes[row, 0].set_xlabel("x"); axes[row, 0].set_ylabel("y")
        color_ax = axes[row, 0].inset_axes([1.025, 0.1, 0.035, 0.8])
        colorbar = fig.colorbar(artist, cax=color_ax)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label(r"$|u|$", fontsize=8, labelpad=2)
        for component, color in enumerate(("#2563eb", "#dc2626")):
            for side, style in (("a", "-"), ("b", "--")):
                axes[row, 1].plot(trace["s"], trace[f"values_{side}"][:, 5, component],
                                  color=color, linestyle=style, linewidth=2 if side == "a" else 1.4,
                                  marker="o" if side == "b" else None, markersize=2.8,
                                  markevery=max(1, len(trace["s"])//12), markerfacecolor="white",
                                  label=f"u{'xy'[component]} · cell {side.upper()}")
        axes[row, 1].set_title("Combined velocity: both edge traces", fontsize=11)
        axes[row, 1].set_xlabel("Common-edge parameter s")
        axes[row, 1].set_ylabel("Velocity component")
        axes[row, 1].grid(alpha=0.2)
        if row == 0:
            axes[row, 1].legend(fontsize=8, ncol=2, loc="best")
        jump, tangent_jump, normalized = metrics(patch)
        for mode in range(6):
            axes[row, 2].semilogy(trace["s"], np.maximum(jump[:, mode], floor),
                                  color=colors[mode], linewidth=1, alpha=0.85, label=MODE_LABELS[mode])
        axes[row, 2].set_title(f"All six modes · {'PASS' if normalized <= tolerance else 'FAIL'}", fontsize=11)
        axes[row, 2].set_xlabel("Common-edge parameter s")
        axes[row, 2].set_ylabel(r"$\max(|[u_x]|, |[u_y]|)$")
        axes[row, 2].grid(alpha=0.18, which="both")
        axes[row, 2].set_ylim(floor*0.7, max(tolerance*2, jump.max()*3, floor*10))
        axes[row, 2].text(0.02, 0.98, f"max value jump: {jump.max():.2e}\nmax tangential jump: {tangent_jump:.2e}",
                          transform=axes[row, 2].transAxes, va="top", fontsize=8,
                          bbox={"fc": "white", "ec": "#e2e8f0", "alpha": 0.92})
    fig.suptitle("BRMixed conformity · independently evaluated neighboring cells", x=0.07, ha="left",
                 fontsize=20, weight="bold", y=0.985)
    fig.text(0.07, 0.952, "Left: combined vector field   |   Middle: velocity traces from A and B   |   Right: jumps of all shared modes",
             fontsize=10.5, color="#475569")
    fig.text(0.07, 0.928, f"Normalized check tolerance: {tolerance:.2e}. Normal derivatives are allowed to jump in an H¹-conforming space.",
             fontsize=10, color="#475569")
    handles, labels = axes[-1, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.54, 0.025), ncol=3, fontsize=9, frameon=False)
    fig.text(0.07, 0.01, f"Log display floor: {floor:.1e}; numerical maxima above are computed from unfloored data.", fontsize=8, color="#64748b")
    save(fig, folder, "conformity")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5_file", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    metadata, data = read_data(args.h5_file)
    folder = args.out_dir or args.h5_file.parent/"figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    maximum = max(np.linalg.norm(data[name]["cells"][0][f"basis_{k:02d}"][:, :2], axis=1).max()
                  for name in CASES[:2] for k in range(12))
    norm = Normalize(0, maximum)
    gallery(data, "rectangular", norm, folder)
    gallery(data, "quadrilateral", norm, folder)
    shared_gallery(data, folder)
    conformity(data, metadata[4], folder)
    success = True
    for name in CASES:
        jump, tangent_jump, normalized = metrics(data[name])
        passed = normalized <= metadata[4]
        success &= passed
        print(f"{name}: max |[u]|={jump.max():.3e}, max |[d_t u]|={tangent_jump:.3e}, "
              f"normalized={normalized:.3e} {'PASS' if passed else 'FAIL'}")
    print(f"Saved four PNG/PDF figures to {folder}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

