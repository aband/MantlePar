#!/usr/bin/env python3
"""Plot actual CSV results. Missing or failed solves are never replaced with zero."""
from __future__ import annotations
import csv
import math
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def resolved(path, config, filename, family, level):
    substitutions = dict(mesh_family=family, level=level, step=0)
    directory = Path(config["output"]["directory"].format(**substitutions))
    if not directory.is_absolute():
        directory = path.parent / directory
    file = Path(filename.format(**substitutions))
    return file if file.is_absolute() else directory / file

def read_row(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Expected one record: {path}")
    return rows[0]

def plot_case(path, config):
    study = config.get("studies", {}).get("convergence", {})
    enabled = study.get("enabled", False)
    families = study["mesh_families"] if enabled else [config["mesh"]["family"]]
    levels = study["levels"] if enabled else 1
    paths = [resolved(path, config, "metrics.csv", f, l) for f in families for l in range(levels)]
    destination = Path(os.path.commonpath([str(p.parent) for p in paths]))
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for family in families:
        previous = None
        for level in range(levels):
            if config["output"].get("solver_report", "csv") != "none":
                report = read_row(resolved(path, config, "solver_report.csv", family, level))
                if report["converged"] != "1":
                    raise ValueError(f"Unconverged result at {family}, level {level}")
            provenance = resolved(path, config, "input_used.yaml", family, level)
            if provenance.exists() and provenance.read_text() != path.read_text():
                raise ValueError(f"Input changed since the run: {provenance}. Rerun before plotting.")
            row = read_row(resolved(path, config, "metrics.csv", family, level))
            for field in list(row):
                if field.startswith("l2_"):
                    error = float(row[field])
                    if not math.isfinite(error) or error < 0:
                        raise ValueError(f"Invalid error value: {field}")
                    row["rate_" + field[3:]] = ""
                    if previous and error > 0 and float(previous[field]) > 0:
                        ratio = float(previous["h"]) / float(row["h"])
                        if ratio > 1:
                            row["rate_" + field[3:]] = math.log(float(previous[field]) / error) / math.log(ratio)
            records.append(row)
            previous = row
    if config["output"].get("convergence_table", "csv") == "csv":
        with (destination / "convergence.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    plots = config["output"].get("plots", {})
    formats = plots.get("formats", ["png", "pdf"])
    if enabled and plots.get("convergence", False):
        names = [key for key in records[0] if key.startswith("l2_")]
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
        for ax, name in zip(axes[0], names):
            for family in families:
                selected = [row for row in records if row["family"] == family]
                h = np.array([float(row["h"]) for row in selected])
                error = np.array([float(row[name]) for row in selected])
                positive = error > 0
                if not np.any(positive):
                    continue
                ax.loglog(h[positive], error[positive], "o-", label=family.replace("_", " "))
            ax.set(xlabel="Maximum cell diameter h", ylabel="L2 error", title=name[3:].replace("_", " "))
            ax.grid(True, which="both", alpha=.25); ax.legend(fontsize=8)
        ref = records[0]["reference"]
        fig.suptitle(config["simulation"]["name"].replace("_", " ") + f" — reference: {ref}")
        fig.tight_layout()
        for ext in formats: fig.savefig(destination / f"convergence.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    profile = config["output"].get("profiles", {})
    if profile.get("enabled", False) and plots.get("profiles", False):
        tables = {}
        for family in families:
            filename = resolved(path, config, profile["file"], family, levels - 1)
            table = np.genfromtxt(filename, delimiter=",", names=True, encoding="utf-8")
            tables[family] = table
        names = [name for name in next(iter(tables.values())).dtype.names if name not in ("distance", "x", "y")]
        columns = min(3, len(names)); rows = math.ceil(len(names) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), squeeze=False)
        for ax, name in zip(axes.flat, names):
            for family, table in tables.items():
                valid = np.isfinite(table[name])
                ax.plot(table[name][valid], table["distance"][valid], label=family.replace("_", " "))
            ax.set(xlabel=name.replace("_", " "), ylabel="Distance along profile")
            ax.grid(True, alpha=.25); ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(names):]: ax.set_visible(False)
        fig.tight_layout()
        for ext in formats: fig.savefig(destination / f"profiles.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(f"Results and plots: {destination}")
