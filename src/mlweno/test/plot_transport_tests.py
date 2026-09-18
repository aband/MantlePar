#!/usr/bin/env python3
"""Plot transport test CSVs. Requires matplotlib; CMake makes plotting optional."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig, destination):
    fig.savefig(destination.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def errors(path, data, output):
    quantities = list(dict.fromkeys(row["quantity"] for row in data))
    jump = "jump" in path.stem
    norms = ("l1", "l2", "linf") if jump else ("l2", "linf")
    fig, axes = plt.subplots(len(quantities), len(norms), squeeze=False,
                             figsize=(4.1 * len(norms), 3.25 * len(quantities)))
    colors = {"all": "#242424", "interior": "#2563a6", "boundary": "#b35c16",
              "cut": "#a3293c", "near_jump": "#b35c16", "smooth_stencil": "#2563a6"}
    labels = {"l1": r"$L^1$ error", "l2": r"$L^2$ error", "linf": r"Maximum error"}
    for i, quantity in enumerate(quantities):
        groups = defaultdict(list)
        for row in data:
            if row["quantity"] == quantity:
                groups[row["region"]].append(row)
        for j, norm in enumerate(norms):
            ax = axes[i, j]
            plotted = False
            for region, values in groups.items():
                pairs = sorted((float(row["h"]), float(row[norm])) for row in values)
                pairs = [(h, e) for h, e in pairs if h > 0 and e > 0 and math.isfinite(e)]
                if pairs:
                    x, y = zip(*pairs)
                    ax.loglog(x, y, "o-", lw=1.5, ms=4, color=colors.get(region),
                              label=region.replace("_", " "))
                    plotted = True
            ax.set_xlabel(r"Mesh spacing $h$")
            ax.set_ylabel(labels[norm])
            ax.set_title(quantity.replace("_", " "))
            ax.grid(True, which="both", alpha=.22)
            if plotted:
                ax.legend(fontsize=8)
            else:
                ax.text(.5, .5, "Errors at zero", ha="center", transform=ax.transAxes)
    fig.suptitle(path.stem.replace("_", " "), fontsize=11)
    fig.tight_layout()
    save(fig, output / (path.stem + "_errors"))


def profile(path, data, output):
    data.sort(key=lambda row: float(row["distance"]))
    x = [float(row["distance"]) for row in data]
    xh = [float(row["distance_over_h"]) for row in data]
    exact = [float(row["exact"]) for row in data]
    reconstructed = [float(row["reconstructed"]) for row in data]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7))
    for ax, horizontal, label in ((axes[0], x, "Signed distance to interface"),
                                   (axes[1], xh, r"Signed distance / $h$")):
        ax.plot(horizontal, exact, color="#242424", lw=1.5, label="Exact")
        ax.plot(horizontal, reconstructed, color="#2563a6", lw=1.2, label="Reconstruction")
        ax.set_xlabel(label)
        ax.set_ylabel("Scalar value")
        ax.grid(alpha=.22)
        ax.legend(fontsize=8)
    axes[0].set_title("Profile")
    axes[1].set_xlim(-4, 4)
    axes[1].set_title("Near the interface")
    for column, label, color in (("weight_large", "Large candidates", "#2563a6"),
                                 ("weight_small", "Small candidates", "#b35c16"),
                                 ("weight_constant", "Constant candidate", "#a3293c")):
        axes[2].plot(xh, [float(row[column]) for row in data], label=label, color=color)
    axes[2].set_xlim(-4, 4)
    axes[2].set_ylim(-.03, 1.03)
    axes[2].set_xlabel(r"Signed distance / $h$")
    axes[2].set_ylabel("Sum of nonlinear weights")
    axes[2].set_title("Candidate selection")
    axes[2].grid(alpha=.22)
    axes[2].legend(fontsize=8)
    fig.suptitle(path.stem.replace("_", " "), fontsize=11)
    fig.tight_layout()
    save(fig, output / path.stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input_dir
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(args.input_dir.glob("*.csv")):
        data = rows(path)
        if not data:
            continue
        if "quantity" in data[0] and "l2" in data[0]:
            errors(path, data, output)
            count += 1
        elif "reconstructed" in data[0] and "distance" in data[0]:
            profile(path, data, output)
            count += 1
    if not count:
        raise SystemExit("No convergence/profile CSV data found")
    print(f"Wrote {count} figures as PNG and PDF to {output}")


if __name__ == "__main__":
    main()
