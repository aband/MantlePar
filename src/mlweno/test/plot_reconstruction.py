#!/usr/bin/env python3
"""Plot reconstruction convergence, constant weights, and the step profile.

python3 plot_reconstruction.py --input-dir build/src/mlweno/reconstruction_results

Writes PNG/PDF figures. Boundary L2 integrates over a shrinking strip.
Reference slopes and fitted rates apply to the complete-stencil interior.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

FIELDS = (("u", r"$u$"), ("ux", r"$\partial_x u$"), ("uy", r"$\partial_y u$"))
BLUE, ORANGE = "#2166ac", "#c95b21"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    return rows


def load_convergence(directory, case, mesh, axis, order, constant):
    path = directory / f"reconstruction_{case}.csv"
    data = {name: [] for name in ("interior", "boundary", "all")}
    settings = None
    for record in read_csv(path):
        if (record["mesh"], int(record["axis"]), int(record["order"]), int(record["constant"])) != (
            mesh, axis, order, constant
        ):
            raise ValueError(f"{path}: metadata mismatch")
        current = tuple(record[k] for k in (
            "field", "seed", "perturbation", "epsilon", "constant_linear_weight", "target_smoothness"
        ))
        if int(record["field"]) != 0:
            raise ValueError(f"{path}: default plots expect the noncritical field")
        if settings is not None and current != settings:
            raise ValueError(f"{path}: refinement settings changed")
        settings = current
        region = record["region"]
        if region not in data:
            raise ValueError(f"{path}: unknown region {region}")
        row = {"N": int(record["N"]), "h": float(record["h"]),
               "epsilon": float(record["epsilon"]),
               "constant_linear_weight": float(record["constant_linear_weight"])}
        expected_h = (2.0 if axis == 1 else 1.0 if axis == 2 else math.sqrt(2)) / row["N"]
        if not math.isclose(row["h"], expected_h, rel_tol=1e-12):
            raise ValueError(f"{path}: unexpected convergence spacing")
        for field, _ in FIELDS:
            for norm in ("l2", "linf"):
                key = f"{field}_{norm}"
                row[key] = float(record[key])
                inactive = (axis == 1 and field == "uy") or (axis == 2 and field == "ux")
                if not math.isfinite(row[key]) or row[key] < 0 or (not inactive and row[key] == 0):
                    raise ValueError(f"{path}: invalid {key}")
        for key in ("constant_mean", "constant_max", "conservation_max"):
            row[key] = float(record[key])
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f"{path}: invalid {key}")
        if row["constant_max"] > 1 + 1e-12:
            raise ValueError(f"{path}: weight exceeds one")
        data[region].append(row)
    levels = None
    for region, rows in data.items():
        rows.sort(key=lambda row: row["N"])
        ns = [row["N"] for row in rows]
        if len(ns) < 3 or len(ns) != len(set(ns)):
            raise ValueError(f"{path}: require at least three unique levels in {region}")
        if levels is not None and levels != ns:
            raise ValueError(f"{path}: regions use different grids")
        levels = ns
    return data


def slope(rows, key):
    xs = [math.log(row["h"]) for row in rows[-3:]]
    ys = [math.log(row[key]) for row in rows[-3:]]
    mx, my = sum(xs) / 3, sum(ys) / 3
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def axis_style(ax, hs):
    ax.set_xlabel(r"Cell spacing $h$")
    ax.grid(True, which="major", color="#d5dbe1", linewidth=0.7)
    ax.grid(True, which="minor", color="#e8ebee", linewidth=0.4)
    ticks = sorted(set(hs))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{h:.3g}" for h in ticks])
    ax.xaxis.set_minor_locator(NullLocator())


def save(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=180)
    plt.close(fig)


def error_panel(ax, data, field, expected):
    interior, boundary = data["interior"], data["boundary"]
    hs = [row["h"] for row in interior]
    for region, rows, color in (("Interior", interior, BLUE), ("Boundary", boundary, ORANGE)):
        for norm, marker, style, text in (("l2", "o", "-", r"$L^2$"), ("linf", "s", "--", "max")):
            key = f"{field}_{norm}"
            suffix = f", rate {slope(rows, key):.2f}" if region == "Interior" else ""
            ax.loglog(hs, [row[key] for row in rows], color=color, marker=marker,
                      linestyle=style, linewidth=1.5, markersize=4,
                      label=f"{region} {text}{suffix}")
    reference = [0.55 * interior[-2][f"{field}_l2"] * (h / hs[-2]) ** expected for h in hs]
    ax.loglog(hs, reference, ":", color="#555555", label=rf"$h^{expected}$ reference")
    axis_style(ax, hs)
    ax.set_ylabel("Absolute error")
    ax.legend(fontsize=8, frameon=False)


def plot_2d(directory, output, mesh, order):
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.4), layout="constrained")
    for constant in (0, 1):
        data = load_convergence(directory, f"{mesh}_r{order}_c{constant}", mesh, 0, order, constant)
        for column, (field, title) in enumerate(FIELDS):
            ax = axes[constant, column]
            error_panel(ax, data, field, order - (field != "u"))
            ax.set_title(f"{title}; constant {'on' if constant else 'off'}")
    fig.suptitle(
        f"ML-WENO ({order},{order-1}) — {mesh} mesh\n"
        "Interior and boundary errors; large reference-square / small target-cell indicators",
        fontsize=13,
    )
    save(fig, output, f"convergence_{mesh}_r{order}")


def plot_1d(directory, output, axis):
    label = "x" if axis == 1 else "y"
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 8.1), layout="constrained")
    for constant in (0, 1):
        data = load_convergence(directory, f"pseudo1d_{label}_c{constant}", "rectangular", axis, 5, constant)
        for column, (field, title) in enumerate((FIELDS[0], FIELDS[axis])):
            ax = axes[constant, column]
            error_panel(ax, data, field, 5 - (field != "u"))
            ax.set_xlabel(rf"Active spacing $\Delta {label}$")
            ax.set_title(f"{title}; constant {'on' if constant else 'off'}")
    fig.suptitle(
        f"Pseudo-1D ML-WENO (5,3), {label} direction\n"
        "One cell in the inactive direction; its derivative is checked separately",
        fontsize=13,
    )
    save(fig, output, f"convergence_pseudo1d_{label}")


def plot_constant_weights(directory, output):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), layout="constrained")
    for row, mesh in enumerate(("rectangular", "quadrilateral")):
        for column, order in enumerate((3, 4)):
            ax = axes[row, column]
            data = load_convergence(directory, f"{mesh}_r{order}_c1", mesh, 0, order, 1)
            hs = [r["h"] for r in data["interior"]]
            for region, color in (("interior", BLUE), ("boundary", ORANGE)):
                for key, style, description in (
                    ("constant_mean", "o-", "area-weighted mean"),
                    ("constant_max", "s--", "maximum"),
                ):
                    points = [(r["h"], r[key]) for r in data[region] if r[key] > 0]
                    if points:
                        ax.loglog(*zip(*points), style, color=color, linewidth=1.5,
                                  label=f"{region}: {description}")
            parameters = data["interior"][0]
            ax.set_title(f"{mesh}, ({order},{order-1})\n"
                         rf"$\epsilon$={parameters['epsilon']:g}, "
                         rf"$d_0$={parameters['constant_linear_weight']:g}")
            ax.set_ylabel(r"Constant weight $\omega_0$")
            axis_style(ax, hs)
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Constant candidate under refinement", fontsize=13)
    save(fig, output, "constant_weights")


def plot_step(directory, output):
    profiles = read_csv(directory / "reconstruction_step.csv")
    weights = read_csv(directory / "reconstruction_step_weights.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), layout="constrained")
    for constant in (0, 1):
        cells = defaultdict(list)
        for record in profiles:
            if int(record["constant"]) == constant:
                cells[int(record["cell"])].append(record)
        for column, limits in enumerate(((0, 2), (0.88, 1.12))):
            ax = axes[constant, column]
            ax.plot([0, 1, 1, 2], [1, 1, 0, 0], "k--", linewidth=1.4, label="Exact step")
            first = True
            for cell in sorted(cells):
                rs = sorted(cells[cell], key=lambda r: float(r["x"]))
                xs, ys = [float(r["x"]) for r in rs], [float(r["u"]) for r in rs]
                if not all(math.isfinite(v) for v in xs + ys):
                    raise ValueError("Nonfinite discontinuity profile")
                # Keep each cell separate to show both traces at shared faces.
                ax.plot(xs, ys, color=BLUE, linewidth=1.4, label="Reconstruction" if first else None)
                first = False
            ax.set_xlim(*limits)
            ax.set_ylim(-0.07, 1.07)
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.set_title(f"{'Full profile' if column == 0 else 'Jump detail'}; constant {'on' if constant else 'off'}")
            ax.grid(color="#e2e6ea")
            ax.legend(fontsize=9, frameon=False)
        ax = axes[constant, 2]
        selected = [r for r in weights if int(r["constant"]) == constant]
        labels = sorted({(r["family"], int(r["candidate"])) for r in selected})
        x_by_cell = {int(r["cell"]): float(r["x"]) for r in selected}
        ns = sorted(x_by_cell)
        for family, candidate in labels:
            values = {int(r["cell"]): float(r["weight"]) for r in selected
                      if r["family"] == family and int(r["candidate"]) == candidate}
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()):
                raise ValueError("Invalid discontinuity weights")
            label = "constant" if family == "constant" else f"{family} {candidate}"
            ax.step([x_by_cell[i] for i in ns], [values.get(i, 0) for i in ns],
                    where="mid", linewidth=1.4, label=label)
        ax.axvline(1, color="#555555", linestyle=":", linewidth=1)
        ax.set_xlim(.85, 1.15)
        ax.set_ylim(-.03, 1.03)
        ax.set_xlabel("Cell-center x")
        ax.set_ylabel("Candidate weight")
        ax.set_title(f"Weights; constant {'on' if constant else 'off'}")
        ax.grid(color="#e2e6ea")
        ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.suptitle("Step reconstruction and nonlinear weights — jump at x = 1", fontsize=13)
    save(fig, output, "discontinuity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "savefig.facecolor": "white"})
    for mesh in ("rectangular", "quadrilateral"):
        for order in (3, 4):
            plot_2d(args.input_dir, output, mesh, order)
    for axis in (1, 2):
        plot_1d(args.input_dir, output, axis)
    plot_constant_weights(args.input_dir, output)
    plot_step(args.input_dir, output)
    print(f"Wrote 8 reconstruction figures (PNG and PDF) to {output}")


if __name__ == "__main__":
    main()
