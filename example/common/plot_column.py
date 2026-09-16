#!/usr/bin/env python3
"""Closed-column plots in the layout of reviewed.pdf Figures 3.11-3.14.

Use --exe to run the existing shared driver through common/run.py first.
Otherwise read existing output only. No simulation parameters live here.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter, ScalarFormatter
import numpy as np
import yaml


class ColumnReference:
    """Eq. (3.153), with the signed vertical force from the input.

    u is physical segregation flux and v=-u. The quadratic solution is the
    same 128-term regular Frobenius branch used by the shared driver.
    Pressure potentials follow from the continuous equations; their one
    common constant is set by q_s(y_top)=0.
    """

    def __init__(self, config):
        flow = config["flow"]
        if flow["system"] != "coupled_stokes_darcy" or flow["parameters"].get("theta", 0) != 0:
            raise ValueError("Column plots require coupled flow with theta=0.")
        if config["porosity"]["source"] != "prescribed_function":
            raise ValueError("The analytic column reference requires prescribed porosity.")
        force = flow["forcing"]["stokes"]
        if force["name"] != "constant" or force["parameters"]["value"][0] != 0:
            raise ValueError("The column reference requires constant vertical forcing.")
        self.gravity = float(force["parameters"]["value"][1])
        self.lower, self.upper = map(float, config["mesh"]["domain"]["y"])
        function = config["porosity"]["prescribed_function"]
        self.kind = function["name"]
        p = function["parameters"]
        self.interface = float(p.get("interface_y", self.upper))
        if self.kind == "constant":
            self.phi0 = float(p["value"])
            self.interface = self.upper
        elif self.kind == "piecewise_constant":
            self.phi0 = float(p["value_below"])
            if p["value_above"] != 0:
                raise ValueError("The step reference requires a dry region above the interface.")
        elif self.kind == "quadratic_below_interface":
            if p["value_above_interface"] != 0:
                raise ValueError("The quadratic reference requires a dry region above the interface.")
            self.a = np.longdouble(p["coefficient"])
            height = self.interface - self.lower
            if not 0 < self.a * height * height <= 0.1:
                raise ValueError("Quadratic reference requires 0 < maximum porosity <= 0.1.")
            self.exponent = (3 + np.sqrt(9 + 4 / self.a)) / 2
            self.part = np.zeros(128, dtype=np.longdouble)
            self.hom = np.zeros(128, dtype=np.longdouble)
            self.hom[0] = 1
            for n in range(2, 128, 2):
                if abs(self.a * n * (n - 3) - 1) <= 1e-12:
                    raise ValueError("The reference does not include resonant logarithmic terms.")
                for coeff, exponent, source in (
                    (self.part, 0, self.a**2 if n == 4 else -self.a**3 if n == 6 else 0),
                    (self.hom, self.exponent, 0),
                ):
                    m = n + exponent
                    rhs = source - self.a**2 / 3 * (m - 2) * (m - 3) * coeff[n - 2]
                    if n >= 4:
                        rhs += 4 * self.a**3 / 3 * (m - 4) * (m - 3) * coeff[n - 4]
                    coeff[n] = rhs / (self.a * m * (m - 3) - 1)
        else:
            raise ValueError(f"Unsupported column porosity: {self.kind}")
        if self.kind != "quadratic_below_interface" and not 0 < self.phi0 < 1:
            raise ValueError("The wet porosity must lie strictly between zero and one.")
        self.pressure_offset = -self._raw(np.array([self.upper]))["qs"][0]

    def porosity(self, y):
        y = np.asarray(y, dtype=float)
        if self.kind == "constant":
            return np.full_like(y, self.phi0)
        if self.kind == "piecewise_constant":
            return np.where(y < self.interface, self.phi0, 0.0)
        return np.where(y < self.interface, float(self.a) * (self.interface - y)**2, 0.0)

    def _raw(self, y):
        y = np.asarray(y, dtype=float)
        phi = self.porosity(y)
        wet = phi > 0
        u = np.zeros_like(y)
        du = np.zeros_like(y)
        height = self.interface - self.lower
        if self.kind != "quadratic_below_interface":
            p = self.phi0
            radius = math.sqrt(3 / (p * (3 + p - 4 * p * p)))
            midpoint = (self.lower + self.interface) / 2
            z = radius * (y[wet] - midpoint)
            half = radius * height / 2
            den = 1 + np.exp(-2 * half)
            cosh_ratio = (np.exp(z - half) + np.exp(-z - half)) / den
            sinh_ratio = (np.exp(z - half) - np.exp(-z - half)) / den
            u[wet] = -p * p * (1 - p) * (1 - cosh_ratio)
            du[wet] = p * p * (1 - p) * radius * sinh_ratio
            phi_integral = p * np.clip(y - self.lower, 0, height)
        else:
            z = np.asarray(self.interface - y[wet], dtype=np.longdouble)
            h = np.longdouble(height)
            poly = np.polynomial.polynomial.polyval
            dpart = np.arange(1, 128, dtype=np.longdouble) * self.part[1:]
            dhom = np.arange(1, 128, dtype=np.longdouble) * self.hom[1:]
            scale = poly(h, self.part) / poly(h, self.hom)
            power = (z / h)**self.exponent
            u[wet] = poly(z, self.part) - scale * power * poly(z, self.hom)
            du[wet] = -poly(z, dpart) + scale * power * (
                self.exponent * poly(z, self.hom) / z + poly(z, dhom))
            distance = np.clip(self.interface - y, 0, height)
            phi_integral = float(self.a) * (height**3 - distance**3) / 3
        u *= self.gravity
        du *= self.gravity
        # F' = gravity*(1-phi), F = q_l + A*u', A=(3+phi-4phi^2)/(3phi).
        total_stress = self.gravity * (y - self.lower - phi_integral)
        qs = total_stress - (1 - 4 * phi) * du / 3
        ql = np.full_like(y, np.nan)
        ql[wet] = total_stress[wet] - (3 + phi[wet] - 4 * phi[wet]**2) * du[wet] / (3 * phi[wet])
        ps = total_stress - 4 * (1 - phi) * du / 3
        return dict(phi=phi, u=u, v=-u, du=du, qs=qs, ql=ql, ps=ps)

    def __call__(self, y):
        result = self._raw(y)
        for name in ("qs", "ql", "ps"):
            result[name] += self.pressure_offset
        return result


def input_path(case):
    path = Path(case).expanduser().resolve()
    return path / "input.yaml" if path.is_dir() else path


def output_path(path, config, name, family, level):
    values = dict(mesh_family=family, level=level, step=0)
    directory = Path(config["output"]["directory"].format(**values))
    if not directory.is_absolute():
        directory = path.parent / directory
    filename = Path(name.format(**values))
    return filename if filename.is_absolute() else directory / filename


def one_row(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Expected one record: {path}")
    return rows[0]


def load_metrics(path, config, families, levels):
    records = []
    height = config["mesh"]["domain"]["y"][1] - config["mesh"]["domain"]["y"][0]
    for family in families:
        previous = None
        for level in range(levels):
            at = lambda name: output_path(path, config, name, family, level)
            report = one_row(at("solver_report.csv"))
            if report["converged"] != "1" or report["subsolvers_converged"] != "1":
                raise ValueError(f"Unconverged solve: {family}, level {level}")
            if at("input_used.yaml").read_bytes() != path.read_bytes():
                raise ValueError(f"Input changed since the run: {at('input_used.yaml')}")
            row = one_row(at("metrics.csv"))
            if row["family"] != family or int(row["level"]) != level:
                raise ValueError(f"Run metadata mismatch: {at('metrics.csv')}")
            row["delta_y"] = height / int(row["ny"])
            for field in ("darcy_segregation_flux", "stokes_velocity"):
                key = "l2_" + field
                if key not in row:
                    continue
                error = float(row[key])
                if not np.isfinite(error) or error < 0:
                    raise ValueError(f"Invalid {key} in {at('metrics.csv')}")
                rate = ""
                if previous and error > 0 and float(previous[key]) > 0:
                    ratio = previous["delta_y"] / row["delta_y"]
                    if ratio > 1:
                        rate = math.log(float(previous[key]) / error) / math.log(ratio)
                row["rate_dy_" + field] = rate
            records.append(row)
            previous = row
    return records


def log_axes(ax):
    for axis, bounds, limits in ((ax.xaxis, ax.dataLim.intervalx, ax.set_xlim),
                                (ax.yaxis, ax.dataLim.intervaly, ax.set_ylim)):
        low, high = bounds
        if np.isfinite(low) and np.isfinite(high) and 0 < low <= high:
            a, b = math.floor(math.log10(low)), math.ceil(math.log10(high))
            if a == b:
                a -= 1
                b += 1
            limits(10.0**a, 10.0**b)
        axis.set_major_locator(LogLocator(base=10, subs=(1,), numticks=12))
        axis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
        axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        axis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", alpha=0.12)


def save_figure(figure, destination, formats):
    for extension in formats:
        figure.savefig(destination.with_suffix("." + extension), dpi=220, bbox_inches="tight")
    plt.close(figure)


def convergence_plot(config, records, families, destination, formats):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for ax, field, label in zip(axes, ("darcy_segregation_flux", "stokes_velocity"),
                               (r"Darcy flux $\boldsymbol{u}$", r"Stokes velocity $\boldsymbol{v}_s$")):
        for i, family in enumerate(families):
            rows = [r for r in records if r["family"] == family]
            dy = np.array([r["delta_y"] for r in rows])
            error = np.array([float(r["l2_" + field]) for r in rows])
            positive = error > 0
            ax.loglog(dy[positive], error[positive], "o-" if i == 0 else "s--",
                      label=family.replace("_", " "))
        log_axes(ax)
        ax.set(xlabel=r"Logical vertical spacing $\Delta y$", ylabel=r"$L^2$ error", title=label)
        ax.legend(fontsize=8)
    fig.suptitle(config["simulation"]["name"].replace("_", " "))
    save_figure(fig, destination / "column_convergence", formats)


def read_fields(filename):
    with h5py.File(filename, "r") as h:
        names = ("cell_centers", "coordinates", "topology", "porosity", "stokes_velocity",
                 "darcy_segregation_flux", "stokes_pressure_assembled", "darcy_pressure_assembled")
        data = {name: np.asarray(h[name]) for name in names}
    corners = data["coordinates"][data["topology"].astype(int), :2]
    x, y = corners[:, :, 0], corners[:, :, 1]
    data["areas"] = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1))
    for name in ("porosity", "stokes_pressure_assembled", "darcy_pressure_assembled"):
        data[name] = data[name].reshape(-1)
    return data


def pressure_potentials(phi, ps, pd, shift):
    wet = phi > 0
    ql = np.full_like(ps, np.nan)
    qs = ps.copy()
    ql[wet] = pd[wet] / np.sqrt(phi[wet])
    qs[wet] = (ps[wet] - np.sqrt(phi[wet]) * pd[wet]) / (1 - phi[wet])
    return qs + shift, ql + shift


def profile_figure(path, config, reference, family, level, metrics, destination, formats):
    profile = config["output"]["profiles"]
    if not profile.get("enabled", False):
        raise ValueError("Column profiles require output.profiles.enabled: true.")
    if profile["start"][0] != profile["end"][0]:
        raise ValueError("Choose a vertical profile line for the column plot.")
    filename = output_path(path, config, profile["file"], family, level)
    samples = np.atleast_1d(np.genfromtxt(filename, delimiter=",", names=True, encoding="utf-8"))
    required = ("y", "porosity", "stokes_velocity_y", "darcy_segregation_flux_y",
                "stokes_pressure_assembled", "darcy_pressure_assembled")
    if any(name not in samples.dtype.names for name in required):
        raise ValueError(f"Required fields missing in {filename}; use the supplied profile.fields.")
    samples = samples[np.argsort(samples["y"])]
    data = read_fields(output_path(path, config, config["output"]["fields"]["file"], family, level))
    centers = data["cell_centers"][:, :2]
    # Apply ONE shared plotting shift to both pressure potentials. The raw
    # exported solution retains its algebraic nullspace gauge.
    exact_ps = reference(centers[:, 1])["ps"]
    shift = np.average(exact_ps - data["stokes_pressure_assembled"], weights=data["areas"])
    qs, ql = pressure_potentials(samples["porosity"], samples["stokes_pressure_assembled"],
                                samples["darcy_pressure_assembled"], shift)
    lower, upper = reference.lower, reference.upper
    y = np.linspace(lower, upper, max(1201, len(samples)))
    if reference.kind != "constant":
        eps = 1e-9 * (upper - lower)
        y = np.unique(np.r_[y, reference.interface - eps, reference.interface + eps])
    exact = reference(y)
    colors = dict(stokes="#d62728", darcy="#1f55d5")
    fig = plt.figure(figsize=(14, 7.8))
    grid = fig.add_gridspec(2, 4, width_ratios=(0.8, 1.2, 1.15, 1.3),
                           left=0.055, right=0.985, bottom=0.12, top=0.87,
                           wspace=0.4, hspace=0.44)
    porosity_ax = fig.add_subplot(grid[:, 0])
    velocity_ax = fig.add_subplot(grid[:, 1])
    darcy_ax = fig.add_subplot(grid[0, 2])
    stokes_ax = fig.add_subplot(grid[1, 2])
    pressure_ax = fig.add_subplot(grid[:, 3])
    for ax in (porosity_ax, velocity_ax, darcy_ax, stokes_ax, pressure_ax):
        ax.set_ylim(lower, upper)
        ax.set_yticks(np.linspace(lower, upper, 5))
        ax.tick_params(direction="in", labelsize=9)
        if reference.kind != "constant":
            ax.axhline(reference.interface, color="0.8", lw=0.7, ls=":", zorder=0)
    porosity_ax.plot(exact["phi"], y, "k-", lw=1.7)
    porosity_ax.set_xlim(0, 1.06 * float(np.max(exact["phi"])))
    porosity_ax.set(title="Porosity", xlabel=r"$\phi$", ylabel=r"Height $y$")
    velocity_ax.plot(exact["v"], y, "k-", lw=1.3, label="Reference Stokes")
    velocity_ax.plot(exact["u"], y, "k-", lw=1.3, label="Reference Darcy")
    velocity_ax.plot(samples["stokes_velocity_y"], samples["y"], "--", color=colors["stokes"], lw=1.6, label=r"Stokes $v$")
    velocity_ax.plot(samples["darcy_segregation_flux_y"], samples["y"], "--", color=colors["darcy"], lw=1.6, label=r"Darcy $u$")
    velocity_ax.set(title="Vertical velocities", xlabel=r"$u,\ v$")
    velocity_ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    nx, ny = int(metrics["nx"]), int(metrics["ny"])
    stride = max(1, math.ceil(ny / 18))
    selected_rows = (np.arange(len(centers)) // nx) % stride == stride // 2
    bounds = config["mesh"]["domain"]["x"]
    for ax, name, title in ((darcy_ax, "darcy_segregation_flux", r"Darcy flux $\boldsymbol{u}$"),
                            (stokes_ax, "stokes_velocity", r"Stokes velocity $\boldsymbol{v}_s$")):
        vectors = data[name][:, :2]
        peak = float(np.max(np.linalg.norm(vectors, axis=1)))
        arrow_length = 0.9 * (upper - lower) * stride / ny
        arrows = ax.quiver(centers[selected_rows, 0], centers[selected_rows, 1],
                           vectors[selected_rows, 0], vectors[selected_rows, 1],
                           angles="xy", scale_units="xy", scale=peak / arrow_length if peak else 1,
                           color="#0072bd", width=0.006, pivot="mid",
                           headlength=4, headaxislength=3.5, minlength=0)
        ax.set_xlim(bounds)
        ax.set_xticks([bounds[0], (bounds[0] + bounds[1]) / 2, bounds[1]])
        ax.set_xlabel(r"$x$")
        ax.set_title(title, fontsize=10, pad=21)
        if peak:
            ax.quiverkey(arrows, 0.40, 1.045, peak, f"{peak:.1e}", angle=90,
                         labelpos="E", fontproperties={"size": 7})
    pressure_ax.plot(exact["qs"], y, color="#00a9ad", lw=1.4, label=r"Reference $q_s$")
    pressure_ax.plot(exact["ql"], y, "k-", lw=1.3, label=r"Reference $q_l$")
    pressure_ax.plot(qs, samples["y"], "--", color=colors["stokes"], lw=1.6, label=r"Computed $q_s$")
    pressure_ax.plot(ql, samples["y"], "--", color=colors["darcy"], lw=1.6, label=r"Computed $q_l$")
    pressure_ax.set(title="Pressure potentials", xlabel=r"$q_s,\ q_l$")
    pressure_ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    for ax in (porosity_ax, velocity_ax, pressure_ax):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 4))
        ax.xaxis.set_major_formatter(formatter)
    fig.suptitle(f"{config['simulation']['name'].replace('_', ' ')} | {family.replace('_', ' ')} | {nx} x {ny}", y=0.97, fontsize=13)
    note = "Solid lines: continuous reference; dashed lines: finite-element profiles. Arrows: sampled cell-center fields."
    if reference.kind != "constant":
        note += "\nLiquid pressure is undefined in dry cells and is left blank."
    fig.text(0.5, 0.035, note, ha="center", fontsize=9)
    save_figure(fig, destination / f"column_{family}_level_{level}", formats)

    exact_samples = reference(samples["y"])
    profile_table = destination / f"column_{family}_level_{level}.csv"
    with profile_table.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["y", "phi_prescribed", "phi_cell_average", "u", "u_reference", "v", "v_reference",
                         "qs", "qs_reference", "ql", "ql_reference", "common_pressure_shift"])
        writer.writerows(zip(samples["y"], exact_samples["phi"], samples["porosity"],
                             samples["darcy_segregation_flux_y"], exact_samples["u"],
                             samples["stokes_velocity_y"], exact_samples["v"], qs, exact_samples["qs"],
                             ql, exact_samples["ql"], np.full(len(samples), shift)))


def plot_case(path, level):
    config = yaml.safe_load(path.read_text())
    reference = ColumnReference(config)
    study = config["studies"]["convergence"]
    families = study["mesh_families"] if study["enabled"] else [config["mesh"]["family"]]
    levels = study["levels"] if study["enabled"] else 1
    shown = levels - 1 if level is None else level
    if not 0 <= shown < levels:
        raise ValueError(f"Choose a level from 0 to {levels - 1}.")
    records = load_metrics(path, config, families, levels)
    paths = [output_path(path, config, "metrics.csv", f, l).parent for f in families for l in range(levels)]
    destination = Path(os.path.commonpath(paths))
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "column_convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plots = config["output"]["plots"]
    formats = plots.get("formats", ["png", "pdf"])
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"}):
        if study["enabled"] and plots.get("convergence", False):
            convergence_plot(config, records, families, destination, formats)
        if plots.get("profiles", False):
            for family in families:
                row = next(r for r in records if r["family"] == family and int(r["level"]) == shown)
                profile_figure(path, config, reference, family, shown, row, destination, formats)
    print(f"Column figures and tables: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", help="Case directories or input.yaml paths")
    parser.add_argument("--level", type=int, help="Profile level; default: finest configured level")
    parser.add_argument("--exe", type=Path, help="Run this shared driver before plotting")
    args = parser.parse_args()
    paths = [input_path(case) for case in args.cases]
    if args.exe:
        runner = Path(__file__).with_name("run.py")
        subprocess.run([sys.executable, str(runner), *map(str, paths), "--exe",
                        str(args.exe.expanduser().resolve()), "--no-plots"], check=True)
    for path in paths:
        plot_case(path, args.level)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, yaml.YAMLError, subprocess.CalledProcessError) as error:
        print(f"Column example: {error}", file=sys.stderr)
        raise SystemExit(1)
