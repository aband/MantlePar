#!/usr/bin/env python3
"""Plot C++ HDivMixed samples without reimplementing the basis in Python.

Usage: python3 plot_hdivmixed.py output/hdivmixed/hdivmixed.h5
Requires numpy, matplotlib and h5py. Writes PNG/PDF figures into figures/
beside the HDF5 file, or into --out-dir. No MPI or ParaView needed to plot.
"""

import argparse
from pathlib import Path
import sys
import tempfile

try:
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
except ImportError as error:
    raise SystemExit(
        "Install the plotting dependencies: python3 -m pip install numpy matplotlib h5py"
    ) from error

CASES = ("rectangular_i", "quadrilateral_i", "rectangular_j", "quadrilateral_j")
MODES = ("shared_linear", "shared_constant", "conforming_velocity")
LABELS = ("Shared linear mode", "Shared constant mode", "Combined field")
COLORS = ("#2563eb", "#c2410c", "#15803d")
SIDES = ("Bottom", "Right", "Top", "Left")


def read_data(path):
    data = {}
    with h5py.File(path, "r") as source:
        metadata = np.asarray(source["metadata"])
        if (metadata.shape != (8,) or not np.isfinite(metadata).all()
                or metadata[0] != 1 or tuple(metadata[6:]) != (8, 3)):
            raise ValueError("Unsupported HDivMixed output schema")
        subdivisions, segments = int(metadata[1]), int(metadata[2])
        if (not 2 <= subdivisions <= 512 or not 2 <= segments <= 100000
                or metadata[1] != subdivisions or metadata[2] != segments
                or metadata[4] <= 0):
            raise ValueError("Invalid HDivMixed sampling metadata")
        count = (subdivisions + 1)**2

        def checked(group, key, shape):
            value = np.asarray(group[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid shape or nonfinite values: {group.name}/{key}")
            return value

        for name in CASES:
            group = source[name]
            cells = []
            for index in range(2):
                g = group[f"cell_{index}"]
                cell = {
                    "coordinates": checked(g, "coordinates", (count, 3)),
                    "corners": checked(g, "corners", (4, 3)),
                    "connectivity": checked(g, "connectivity", (subdivisions**2, 4)),
                    "coefficients": checked(g, "coefficients", (8,)),
                    "pressure_basis": checked(g, "pressure_basis", (count,)),
                }
                for field in [f"basis_{k:02d}" for k in range(8)] + list(MODES):
                    cell[field] = checked(g, field, (count, 3))
                    cell[f"{field}_divergence"] = checked(g, f"{field}_divergence", (count,))
                cells.append(cell)
            g = group["trace"]
            trace = {
                "s": checked(g, "s", (segments+1,)),
                "coordinates": checked(g, "coordinates", (segments+1, 3)),
                "normals": checked(g, "normals", (2, 3)),
                "common_normal": checked(g, "common_normal", (3,)),
                "common_tangent": checked(g, "common_tangent", (3,)),
                "expected_normal": checked(g, "expected_normal", (segments+1, 3)),
            }
            for side in ("a", "b"):
                trace[f"values_{side}"] = checked(g, f"values_{side}", (segments+1, 3, 3))
                for field in ("normal", "tangent", "divergence"):
                    trace[f"{field}_{side}"] = checked(g, f"{field}_{side}", (segments+1, 3))
            data[name] = {"cells": cells, "trace": trace,
                          "dofs": checked(group, "shared_local_dofs", (2, 2)).astype(int)}
    return metadata, data


def outline(ax, corners, **kwargs):
    closed = np.concatenate((corners, corners[:1]), axis=0)
    ax.plot(closed[:, 0], closed[:, 1], **kwargs)


def vector_field(ax, cell, name, norm, arrows=6):
    count = int(round(np.sqrt(len(cell["coordinates"]))))
    xyz = cell["coordinates"].reshape(count, count, 3)
    field = cell[name].reshape(count, count, 3)
    magnitude = np.linalg.norm(field[..., :2], axis=-1)
    artist = ax.pcolormesh(xyz[..., 0], xyz[..., 1], magnitude,
                          shading="gouraud", cmap="viridis", norm=norm, rasterized=True)
    indices = np.unique(np.linspace(1, count-2, min(arrows, count-2)).astype(int))
    selection = np.ix_(indices, indices)
    span = max(np.ptp(cell["corners"][:, 0]), np.ptp(cell["corners"][:, 1]))
    ax.quiver(xyz[..., 0][selection], xyz[..., 1][selection],
              field[..., 0][selection], field[..., 1][selection],
              angles="xy", scale_units="xy", scale=max(norm.vmax, 1e-15)/(0.13*span),
              color="white", width=0.006, headwidth=3.8, headlength=4.3, pivot="middle")
    outline(ax, cell["corners"], color="#17202a", linewidth=1.2, zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)
    return artist


def mark_patch(ax, patch):
    for index, cell in enumerate(patch["cells"]):
        center = cell["corners"][:, :2].mean(axis=0)
        ax.text(*center, "AB"[index], ha="center", va="center", fontsize=9,
                bbox={"fc": "white", "ec": "none", "alpha": 0.9}, zorder=6)
    edge = patch["trace"]["coordinates"][[0, -1], :2]
    ax.plot(edge[:, 0], edge[:, 1], color="#fbbf24", linewidth=2.3, linestyle="--", zorder=5)
    middle = edge.mean(axis=0)
    normal = patch["trace"]["common_normal"][:2]
    length = np.linalg.norm(edge[1]-edge[0])
    ax.annotate("", middle+0.2*length*normal, middle,
                arrowprops={"arrowstyle": "->", "color": "#b91c1c", "lw": 2}, zorder=7)
    ax.text(*(middle+0.23*length*normal), "n", color="#b91c1c", fontsize=10, weight="bold",
            bbox={"fc": "white", "ec": "none", "alpha": 0.85}, zorder=7)


def save(fig, folder, stem):
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
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    fig.subplots_adjust(left=0.065, right=0.90, bottom=0.10, top=0.81, wspace=0.30, hspace=0.48)
    for k, ax in enumerate(axes.flat):
        artist = vector_field(ax, cell, f"basis_{k:02d}", norm)
        divergence = cell[f"basis_{k:02d}_divergence"][0]
        ax.set_title(rf"$\psi_{{{k}}}$ · {SIDES[k%4]}" + f"\ndiv = {divergence:.4g}", fontsize=10.5, pad=8)
        if k >= 4:
            ax.set_xlabel("x", fontsize=10)
        if k % 4 == 0:
            ax.set_ylabel(("Linear trace" if k < 4 else "Constant trace")+"\ny", fontsize=10)
    color_ax = fig.add_axes([0.928, 0.17, 0.016, 0.56])
    fig.colorbar(artist, cax=color_ax, label=r"Basis magnitude $|\psi_k|$")
    fig.suptitle(f"HDivMixed · {shape} element", x=0.065, ha="left", fontsize=20, weight="bold", y=0.975)
    fig.text(0.065, 0.903, "Eight local vector basis functions · arrows show the vector field · color shows its magnitude",
             fontsize=11, color="#475569")
    fig.text(0.065, 0.861, "Top row: linear normal trace and zero divergence. Bottom row: constant normal trace and constant divergence.",
             fontsize=10, color="#475569")
    save(fig, folder, f"basis_{shape}")


def shared_gallery(data, direction, folder):
    cases = [f"{shape}_{direction}" for shape in ("rectangular", "quadrilateral")]
    norms = [Normalize(0, max(np.linalg.norm(cell[mode][:, :2], axis=1).max()
                             for name in cases for cell in data[name]["cells"]))
             for mode in MODES]
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.8 if direction == "j" else 6.6))
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.10, top=0.80, wspace=0.45, hspace=0.55)
    for row, name in enumerate(cases):
        patch = data[name]
        for mode, ax in enumerate(axes[row]):
            for cell in patch["cells"]:
                artist = vector_field(ax, cell, MODES[mode], norms[mode], arrows=5)
            mark_patch(ax, patch)
            subtitle = "All eight local modes" if mode == 2 else (
                f"A: {patch['dofs'][0, mode]}   ↔   B: {patch['dofs'][1, mode]}")
            ax.set_title(f"{LABELS[mode]}\n{subtitle}", fontsize=11)
            ax.set_xlabel("x", fontsize=9)
            if mode == 0:
                ax.set_ylabel(f"{name.split('_')[0].capitalize()}\ny", fontsize=10)
            color_ax = ax.inset_axes([1.03, 0.10, 0.03, 0.80])
            fig.colorbar(artist, cax=color_ax).ax.tick_params(labelsize=7)
    fig.suptitle(f"HDivMixed · two neighbors in the {direction} direction",
                 x=0.07, ha="left", fontsize=20, weight="bold", y=0.975)
    fig.text(0.07, 0.913, "Top: rectangular cells   |   Bottom: perturbed quadrilaterals   |   Dashed gold: shared edge",
             fontsize=10.5, color="#475569")
    fig.text(0.07, 0.867, "The red arrow defines the same normal on both cells. Normal components match; tangential components may jump.",
             fontsize=10, color="#475569")
    save(fig, folder, f"shared_basis_{direction}")


def metrics(patch):
    t = patch["trace"]
    # Recompute projections from the exported vectors, rather than relying on
    # the C++ summary. Tangent/divergence jumps never enter the pass criterion.
    a = t["values_a"] @ t["common_normal"]
    b = t["values_b"] @ t["common_normal"]
    tangent_a = t["values_a"] @ t["common_tangent"]
    tangent_b = t["values_b"] @ t["common_tangent"]
    jump = np.abs(a-b)
    normalized_jump = jump/np.maximum(1, np.maximum(np.abs(a), np.abs(b)))
    expected = t["expected_normal"]
    expected_error = max(
        np.max(np.abs(value-expected)/np.maximum(1, np.maximum(np.abs(value), np.abs(expected))))
        for value in (a, b))
    normal_error = np.max(np.abs(t["normals"]-t["common_normal"]))
    score = max(normalized_jump.max(), expected_error, normal_error)
    return a, b, tangent_a, tangent_b, jump, normalized_jump, score


def conformity(data, tolerance, folder):
    fig, axes = plt.subplots(4, 3, figsize=(14.3, 13.8))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.105, top=0.88, wspace=0.35, hspace=0.62)
    floor = max(1e-18, tolerance*1e-7)
    for row, name in enumerate(CASES):
        trace = data[name]["trace"]
        a, b, ta, tb, raw_jump, jump, score = metrics(data[name])
        s = trace["s"]
        for mode, color in enumerate(COLORS):
            axes[row, 0].plot(s, a[:, mode], color=color, linewidth=2)
            axes[row, 0].plot(s, b[:, mode], color=color, linewidth=1.2, linestyle="--",
                              marker="o", markersize=3, markevery=max(1, len(s)//12), markerfacecolor="white")
            axes[row, 1].semilogy(s, np.maximum(jump[:, mode], floor), color=color, linewidth=1, alpha=0.85)
        shape, direction = name.split("_")
        axes[row, 0].set_title(f"{shape.capitalize()} · {direction} neighbors", fontsize=11, weight="bold")
        axes[row, 0].set_ylabel(r"Normal component $u\cdot n$")
        axes[row, 1].axhline(tolerance, color="#64748b", linestyle=":", linewidth=1.4)
        axes[row, 1].set_title(f"Normal continuity · {'PASS' if score <= tolerance else 'FAIL'}", fontsize=11)
        axes[row, 1].set_ylabel("Normalized normal jump")
        axes[row, 1].set_ylim(floor*0.7, max(tolerance*8, jump.max()*4))
        axes[row, 1].text(0.03, 0.95, f"max |[u·n]| = {raw_jump.max():.2e}",
                          transform=axes[row, 1].transAxes, va="top", fontsize=8.5,
                          bbox={"fc": "white", "ec": "#e2e8f0", "alpha": 0.95})
        axes[row, 2].plot(s, ta[:, 2], color="#2563eb", linewidth=2, label="Cell A")
        axes[row, 2].plot(s, tb[:, 2], color="#c2410c", linewidth=1.8, linestyle="--", label="Cell B")
        axes[row, 2].set_title("Combined field · tangential component", fontsize=10.5)
        axes[row, 2].set_ylabel(r"$u\cdot t$ (jumps allowed)")
        axes[row, 2].text(0.03, 0.95, f"max |[u·t]| = {np.max(np.abs(ta[:, 2]-tb[:, 2])):.2e}",
                          transform=axes[row, 2].transAxes, va="top", fontsize=8.5,
                          bbox={"fc": "white", "ec": "#e2e8f0", "alpha": 0.95})
        axes[row, 2].margins(y=0.2)
        if row == 0:
            axes[row, 2].legend(fontsize=8, loc="lower right")
        for ax in axes[row]:
            ax.set_xlabel("Shared-edge parameter s")
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=8)
    fig.suptitle("HDivMixed conformity · compare the normal component", x=0.07, ha="left",
                 fontsize=20, weight="bold", y=0.975)
    fig.text(0.07, 0.940, "Left: normal traces (solid A, dashed/circles B)   |   Middle: normal jumps   |   Right: tangential traces",
             fontsize=10.5, color="#475569")
    fig.text(0.07, 0.916, f"Tolerance: {tolerance:.2e}. Checks include the prescribed linear/constant traces. Tangential and divergence jumps are allowed.",
             fontsize=9.5, color="#475569")
    handles = [Line2D([], [], color=color, lw=2, label=label) for color, label in zip(COLORS, LABELS)]
    handles.append(Line2D([], [], color="#64748b", ls=":", label="Tolerance"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.039), ncol=4,
               fontsize=10, frameon=False)
    fig.text(0.07, 0.019, f"Log display floor: {floor:.1e}; reported maxima use unfloored values. Each cell is evaluated independently.",
             fontsize=9, color="#64748b")
    save(fig, folder, "conformity")


def divergence_gallery(data, folder):
    # Show the two matched modes and combined field on both sides. The constant
    # mode's divergence generally differs between the two quadrilaterals.
    names = ("rectangular_i", "quadrilateral_i")
    maximum = max(np.max(np.abs(cell[f"{mode}_divergence"]))
                  for name in names for cell in data[name]["cells"] for mode in MODES)
    norm = Normalize(-max(maximum, 1e-15), max(maximum, 1e-15))
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.4))
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.10, top=0.80, wspace=0.30, hspace=0.52)
    for row, name in enumerate(names):
        patch = data[name]
        for mode, ax in enumerate(axes[row]):
            values = []
            for cell in patch["cells"]:
                count = int(round(np.sqrt(len(cell["coordinates"]))))
                xyz = cell["coordinates"].reshape(count, count, 3)
                scalar = cell[f"{MODES[mode]}_divergence"].reshape(count, count)
                artist = ax.pcolormesh(xyz[..., 0], xyz[..., 1], scalar,
                                      shading="gouraud", norm=norm, cmap="coolwarm", rasterized=True)
                outline(ax, cell["corners"], color="#17202a", linewidth=1.2)
                values.append(scalar[0, 0])
            mark_patch(ax, patch)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{LABELS[mode]}\nA: {values[0]:.4g}   B: {values[1]:.4g}", fontsize=10.5)
            ax.set_xlabel("x", fontsize=9)
            ax.tick_params(labelsize=8)
            if mode == 0:
                ax.set_ylabel(f"{name.split('_')[0].capitalize()}\ny", fontsize=10)
    color_ax = fig.add_axes([0.925, 0.18, 0.015, 0.53])
    fig.colorbar(artist, cax=color_ax, label=r"Physical divergence $\nabla\cdot u$")
    fig.suptitle("Cellwise divergence on neighboring elements", x=0.07, ha="left",
                 fontsize=20, weight="bold", y=0.975)
    fig.text(0.07, 0.908, "Linear modes have zero divergence. Constant modes and their combinations have constant divergence within each cell.",
             fontsize=10, color="#475569")
    fig.text(0.07, 0.858, "Different values across the shared edge are compatible with H(div) conformity.",
             fontsize=10.5, color="#475569")
    save(fig, folder, "divergence")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5_file", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    metadata, data = read_data(args.h5_file)
    folder = args.out_dir or args.h5_file.parent/"figures"
    folder.mkdir(parents=True, exist_ok=True)
    maximum = max(np.linalg.norm(data[f"{shape}_i"]["cells"][0][f"basis_{k:02d}"][:, :2], axis=1).max()
                  for shape in ("rectangular", "quadrilateral") for k in range(8))
    norm = Normalize(0, max(maximum, 1e-15))
    for shape in ("rectangular", "quadrilateral"):
        gallery(data, shape, norm, folder)
    for direction in ("i", "j"):
        shared_gallery(data, direction, folder)
    conformity(data, metadata[4], folder)
    divergence_gallery(data, folder)
    passed = True
    for name in CASES:
        *_, score = metrics(data[name])
        ok = score <= metadata[4]
        passed = passed and ok
        print(f"{name}: normalized normal-trace discrepancy {score:.3e} ({'PASS' if ok else 'FAIL'})")
    print(f"Wrote six PNG/PDF figures to {folder}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"HDivMixed plotting failed: {error}") from error
