#!/usr/bin/env python3
"""Plot Q1/Q2 function and derivative convergence from the four test CSVs.

python3 plot_tensorstencilpoly.py --input-dir path/to/tensorstencilpoly_results

Outputs rectangular_convergence.{png,pdf} and
quadrilateral_convergence.{png,pdf}. Requires matplotlib.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

FIELDS = (("u", r"$u$"), ("ux", r"$\partial_x u$"), ("uy", r"$\partial_y u$"))


def read_case(directory, mesh, degree):
    path = directory / f"tensorstencilpoly_{mesh}_q{degree}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if len(records) < 3:
        raise ValueError(f"{path}: at least three refinement levels are required")
    records.sort(key=lambda record: int(record["N"]))
    rows = []
    seen = set()
    settings = None
    for record in records:
        n = int(record["N"])
        if record["mesh"] != mesh or int(record["degree"]) != degree or n in seen:
            raise ValueError(f"{path}: inconsistent mesh/degree or repeated N")
        seen.add(n)
        current_settings = (int(record["seed"]), float(record["perturbation"]))
        if settings is not None and current_settings != settings:
            raise ValueError(f"{path}: mesh settings changed between levels")
        settings = current_settings
        row = {"N": n, "h": float(record["h"])}
        if not math.isclose(row["h"], math.sqrt(2.0) / n, rel_tol=1e-12):
            raise ValueError(f"{path}: h disagrees with the test's domain and N")
        for field, _ in FIELDS:
            for norm in ("l2", "linf"):
                key = f"{field}_{norm}"
                row[key] = float(record[key])
                if not math.isfinite(row[key]) or row[key] <= 0:
                    raise ValueError(f"{path}: {key} must be finite and positive")
        rows.append(row)
    return rows, settings


def fitted_rate(h, errors):
    x = [math.log(value) for value in h[-3:]]
    y = [math.log(value) for value in errors[-3:]]
    mx, my = sum(x) / 3, sum(y) / 3
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum(
        (a - mx) ** 2 for a in x
    )


def plot_mesh(directory, output, mesh):
    cases = {degree: read_case(directory, mesh, degree) for degree in (1, 2)}
    if cases[1][1] != cases[2][1]:
        raise ValueError(f"{mesh}: Q1 and Q2 use different mesh settings")
    seed, perturbation = cases[1][1]
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.6), layout="constrained")
    for row_index, degree in enumerate((1, 2)):
        rows, _ = cases[degree]
        h = [row["h"] for row in rows]
        for column, (field, title) in enumerate(FIELDS):
            ax = axes[row_index, column]
            l2 = [row[f"{field}_l2"] for row in rows]
            maximum = [row[f"{field}_linf"] for row in rows]
            expected = degree + 1 if field == "u" else degree
            ax.loglog(h, l2, "o-", color="#2166ac", linewidth=1.8,
                      label=rf"$L^2$ (rate {fitted_rate(h, l2):.2f})")
            ax.loglog(h, maximum, "s-", color="#c95b21", linewidth=1.8,
                      label=rf"Sampled max (rate {fitted_rate(h, maximum):.2f})")
            reference = [0.6 * l2[-2] * (value / h[-2]) ** expected for value in h]
            ax.loglog(h, reference, "--", color="#555555", linewidth=1.3,
                      label=rf"$h^{expected}$ reference")
            ax.set_title(rf"$Q_{degree}$: {title}")
            ax.set_xlabel(r"Nominal cell size $h$")
            ax.set_ylabel("Absolute error")
            ax.grid(True, which="major", color="#d5dbe1", linewidth=0.7)
            ax.grid(True, which="minor", color="#e8ebee", linewidth=0.4)
            ax.set_xticks(sorted(h))
            ax.set_xticklabels([f"{value:.3g}" for value in sorted(h)])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.legend(loc="best", frameon=False)
    if mesh == "rectangular":
        detail = "Uniform rectangles, aspect ratio 2:1"
    else:
        detail = f"Convex quadrilaterals, {100 * perturbation:g}% vertex perturbation, seed {seed}"
    fig.suptitle(
        f"Tensor polynomial convergence — {detail}\n"
        r"$[0,2]\times[0,1]$; physical derivatives; full stencils at boundaries",
        fontsize=13,
    )
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output / f"{mesh}_convergence.{extension}"
        fig.savefig(path, dpi=180)
        print(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input_dir
    for mesh in ("rectangular", "quadrilateral"):
        plot_mesh(args.input_dir, output, mesh)


if __name__ == "__main__":
    main()
