#!/usr/bin/env python3
"""Render recorded cell-center porosity as an MP4, without interpolating fields."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mantlepar-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from movie_fields import SpatialGrid, frame_indices


def local_file(directory, name):
    path = (directory / name).resolve()
    if path.parent != directory:
        raise ValueError("Snapshot files must be in the output directory")
    return path


def load_series(directory):
    directory = Path(directory).resolve()
    series = json.loads((directory / "porosity_series.json").read_text())
    final = json.loads((directory / "visualization.json").read_text())
    if series.get("status") != "complete" or series.get("schema_version") != 1:
        raise ValueError("A completed run with -couple_porosity_snapshots is required")
    if final.get("status") != "complete" or final.get("schema_version") != 2:
        raise ValueError("The final visualization must be complete")
    if series.get("sampling") != "cell centers" or series.get("units") != "fraction":
        raise ValueError("Expected cell-center porosity in fraction units")
    count, frames = series["cell_count"], series["frames"]
    if count != final["cell_count"] or frames != final["accepted_steps"] + 1:
        raise ValueError("Snapshot dimensions differ from the completed evolution")
    if len(series["files"]) != final["ranks"] or len(set(series["files"])) != final["ranks"]:
        raise ValueError("Snapshot manifest must list each MPI rank exactly once")
    phi = np.empty((frames, count))
    seen = np.zeros(count, dtype=int)
    clock = None
    for name in series["files"]:
        path = local_file(directory, name)
        with path.open() as stream:
            columns = next(csv.reader(stream))
        if columns[:3] != ["step", "time", "time_years"] or any(not c.startswith("phi_") for c in columns[3:]):
            raise ValueError("Unexpected snapshot columns")
        ids = np.array([int(c[4:]) for c in columns[3:]], dtype=int)
        if (ids < 0).any() or (ids >= count).any() or len(set(ids)) != len(ids):
            raise ValueError("Invalid or repeated snapshot cell IDs")
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if data.shape != (frames, len(columns)) or not np.isfinite(data).all():
            raise ValueError("Incomplete or nonfinite snapshot data")
        if clock is None:
            clock = data[:, :3]
        elif not np.array_equal(clock, data[:, :3]):
            raise ValueError("MPI snapshot times do not agree")
        seen[ids] += 1
        phi[:, ids] = data[:, 3:]
    if not (seen == 1).all() or not np.array_equal(clock[:, 0], np.arange(frames)):
        raise ValueError("Missing cells or accepted time steps")
    if frames < 2 or not (np.diff(clock[:, 1]) > 0).all() or phi.min() < 0 or phi.max() >= 1:
        raise ValueError("Need at least two increasing times with admissible porosity")
    np.testing.assert_allclose(clock[:, 2], clock[:, 1] * series["time_scale_years"], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(clock[[0, -1], 1], [series["time_start"], series["time_end"]], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(clock[-1, 1], final["time"], rtol=1e-14, atol=1e-14)
    polygons = np.empty((count, 4, 2))
    mesh_seen = np.zeros(count, dtype=int)
    center_seen = np.zeros(count, dtype=int)
    for entry in final["files"]:
        mesh = np.atleast_1d(np.genfromtxt(local_file(directory, entry["mesh"]), delimiter=",", names=True))
        ids = mesh["cell_id"].astype(int)
        if (ids < 0).any() or (ids >= count).any() or len(set(ids)) != len(ids):
            raise ValueError("Invalid mesh cell IDs")
        mesh_seen[ids] += 1
        for k in range(4):
            polygons[ids, k, 0] = mesh[f"x{k}"] * final["scales"]["length_m"] / 1000
            polygons[ids, k, 1] = mesh[f"y{k}"] * final["scales"]["length_m"] / 1000
        centers = np.atleast_1d(np.genfromtxt(local_file(directory, entry["centers"]), delimiter=",", names=True))
        ids = centers["entity_id"].astype(int)
        if (ids < 0).any() or (ids >= count).any() or len(set(ids)) != len(ids):
            raise ValueError("Invalid final center cell IDs")
        center_seen[ids] += 1
        np.testing.assert_allclose(phi[-1, ids], centers["phi"], rtol=1e-12, atol=1e-14)
    if not (mesh_seen == 1).all() or not (center_seen == 1).all() or not np.isfinite(polygons).all():
        raise ValueError("Incomplete mesh or final center data")
    return final, clock, phi, polygons


def render(directory, output, seconds=12, fps=24, porosity_max=None):
    final, clock, phi, polygons = load_series(directory)
    if not np.isfinite(seconds) or seconds <= 0 or fps < 1:
        raise ValueError("Duration and frame rate must be positive")
    maximum = max(0.01, math.ceil(float(phi.max()) * 100) / 100) if porosity_max is None else porosity_max
    if not np.isfinite(maximum) or maximum < phi.max() or maximum <= 0:
        raise ValueError("Porosity color limit must include every recorded value")
    delta = 100 * (phi - phi[0])  # Percentage-point change, not relative percent change.
    change_max = max(0.1, math.ceil(float(np.abs(delta).max()) * 10) / 10)
    # Quantize physical timestamps to video frames, reserving at least one frame
    # per state (including a very short last time step). Increase duration when
    # necessary instead of dropping snapshots or interpolating the fields.
    indices = frame_indices(clock, seconds, fps)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension")
    if not FFMpegWriter.isAvailable():
        raise ValueError("ffmpeg is required to encode the MP4")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100)
    fig.subplots_adjust(left=.065, right=.945, bottom=.16, top=.79, wspace=.34)
    fig.suptitle(f"Porosity evolution · {final['nx']} × {final['ny']}", y=.972, fontsize=23, fontweight="bold")
    stamp = fig.text(.5, .852, "", ha="center", va="center", fontsize=15)
    fig.text(.5, .906, "Phase-coupled H/C transport and Darcy–Stokes flow", ha="center", va="center", fontsize=12, color="#555555")
    collections = []
    domain = [polygons[:,:,0].min(),polygons[:,:,0].max(),polygons[:,:,1].min(),polygons[:,:,1].max()]
    grid = SpatialGrid(polygons.mean(axis=1),domain,resolution=180)
    for ax, values, cmap, limits, title, label in zip(
        axes, [phi[0], delta[0]], ["viridis", "RdBu_r"], [(0, maximum), (-change_max, change_max)],
        ["Porosity φ", "Change from the preheat state"], ["Liquid fraction", "Δφ (percentage points)"]
    ):
        collection = ax.imshow(grid.sample(values),extent=domain,origin='lower',cmap=cmap,
                               vmin=limits[0],vmax=limits[1],interpolation='nearest')
        ax.set_xlim(polygons[:, :, 0].min(), polygons[:, :, 0].max())
        ax.set_ylim(polygons[:, :, 1].min(), polygons[:, :, 1].max())
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=15, pad=12)
        ax.set_xlabel("x (km)", fontsize=12)
        ax.set_ylabel("y (km)", fontsize=12)
        fig.colorbar(collection, ax=ax, fraction=.048, pad=.035).set_label(label, fontsize=11)
        collections.append(collection)
    fig.text(.5, .072, "Cell-center phase values · Fixed color scales · Recorded accepted states", ha="center", fontsize=11, color="#444444")
    fig.text(.5, .038, "Elapsed time starts after preheat; adaptive steps are paced by physical time.", ha="center", fontsize=10, color="#666666")

    def draw(index):
        collections[0].set_data(grid.sample(phi[index]))
        collections[1].set_data(grid.sample(delta[index]))
        stamp.set_text(f"Elapsed: {clock[index, 2]:.2f} years    |    t = {clock[index, 1]:.5f}    |    Step {int(clock[index, 0])}")

    temporary = output.with_name(output.stem + ".tmp.mp4")
    writer = FFMpegWriter(fps=fps, codec="libx264", metadata={"title": f"Porosity evolution {final['nx']}x{final['ny']}"},
                         extra_args=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2"])
    with writer.saving(fig, str(temporary), dpi=100):
        for index in indices:
            draw(index)
            writer.grab_frame(facecolor="white")
    temporary.replace(output)
    draw(len(clock) - 1)
    poster = output.with_name(output.stem + "_poster.png")
    fig.savefig(poster, dpi=100, facecolor="white")
    plt.close(fig)
    details = {
        "source_directory": str(Path(directory).resolve()), "sampling": "cell centers", "nx": final["nx"], "ny": final["ny"],
        "recorded_states": len(clock), "video_frames": len(indices), "fps": fps,
        "duration_seconds": len(indices) / fps, "time_start": float(clock[0, 1]), "time_end": float(clock[-1, 1]),
        "years_start": float(clock[0, 2]), "years_end": float(clock[-1, 2]),
        "porosity_color_range": [0, maximum], "change_color_range_percentage_points": [-change_max, change_max],
        "playback": "Quantize physical timestamps to video frames, reserving at least one frame per accepted state; one-second endpoint pauses; no field interpolation",
        "state_index_per_video_frame": indices.tolist(),
    }
    output.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
    output.with_suffix(".html").write_text(f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Porosity evolution — {final['nx']} × {final['ny']}</title>
<style>body{{max-width:1280px;margin:32px auto;padding:0 20px;font:17px/1.6 system-ui;color:#19212d;background:#f5f7fa}}video{{width:100%;background:white;border-radius:10px}}h1{{font-size:28px}}a{{color:#1760a5}}</style>
<h1>Porosity evolution · {final['nx']} × {final['ny']}</h1>
<p>t = {clock[0,1]:g}–{clock[-1,1]:g}; {clock[-1,2]-clock[0,2]:.2f} years after preheat. {len(clock)} recorded states.</p>
<video controls autoplay muted loop playsinline preload="metadata" poster="{html.escape(poster.name)}"><source src="{html.escape(output.name)}" type="video/mp4"></video>
<p>Porosity is evaluated at cell centers. Both color scales stay fixed. The right panel shows the change in liquid fraction in percentage points. Snapshot times are quantized to video frames, with at least one frame per saved state; fields are not interpolated.</p>
<p><a href="{html.escape(output.name)}" download>Download MP4</a> · <a href="{html.escape(output.with_suffix('.json').name)}">Frame timing and provenance</a></p></html>''')
    print(f"Video: {output}\nPlayer: {output.with_suffix('.html')}\n{len(clock)} recorded states, {len(indices)} video frames, {len(indices)/fps:g} seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=float, default=12, help="Playback duration, extended if needed to show every state, plus one-second endpoint pauses")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--porosity-max", type=float, help="Fixed color limit; must cover all samples")
    args = parser.parse_args()
    try:
        render(args.input_dir, args.output or args.input_dir / "visualization/porosity_evolution.mp4",
               args.seconds, args.fps, args.porosity_max)
    except (ValueError, OSError, KeyError, AssertionError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
