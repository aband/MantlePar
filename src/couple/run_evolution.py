#!/usr/bin/env python3
"""Build and run the corner case, optionally preheating first, then make videos."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
# Geometry of the corner case templates; the melt outlet is independent of the
# thermal warm strip. Match their require_boundary_vertex tolerance.
DOMAIN_X_END = 0.5
MELT_OUTLET_END = 0.05
BOUNDARY_TOLERANCE = 1e-12


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mesh = parser.add_mutually_exclusive_group()
    mesh.add_argument('--grid', type=int, default=20, metavar='N', help='Square mesh N x N (default: 20)')
    mesh.add_argument('--mesh', type=int, nargs=2, metavar=('NX', 'NY'), help='Rectangular mesh; boundary endpoints must lie on mesh vertices, ny >= 5')
    parser.add_argument('--warm-end', type=float, default=0.025, metavar='X',
                        help='Warm top boundary endpoint and solid-velocity ramp cutoff in both stages; nondimensional, 0 < X <= 0.5 (default: 0.025)')
    parser.add_argument('--run-preheat', action='store_true', help='Run fresh dry preheat first (enabled by run.sh; evolve.sh reuses existing preheat)')
    parser.add_argument('--final-time', '--end-time', dest='end_time', type=float, default=1,
                        help='Full evolution final time, measured from zero after preheat; nondimensional (default: 1)')
    parser.add_argument('--preheat-final-time', '--preheat-end-time', dest='preheat_end_time', type=float, default=15000,
                        help='Preheat final time; nondimensional (default: 15000)')
    parser.add_argument('--maximum-steps', type=int, default=20000, help='Evolution step limit (default: 20000)')
    parser.add_argument('--preheat-maximum-steps', type=int, default=1000000, help='Preheat step limit (default: 1000000)')
    parser.add_argument('--preheat', type=Path, help='Preheat output directory for run.sh; completed input directory for evolve.sh')
    parser.add_argument('--output', type=Path, help='Full evolution output directory')
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build/evolution')
    parser.add_argument('--hdf5-root', type=Path, default=Path(os.environ.get('HDF5_ROOT', '/home/renpo/system/hdf5-install')))
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=12)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--render-only', action='store_true', help='Render existing evolution snapshots, including the combined video; skip both simulations')
    parser.add_argument('--report-only', action='store_true', help='Generate the conditions PDF from a completed run; skip simulation and movies')
    parser.add_argument('--dry-run', action='store_true', help='Print paths and commands; do not write, build, run, or render')
    args = parser.parse_args(argv)
    args.nx, args.ny = args.mesh or (args.grid, args.grid)
    if args.nx < 1 or args.ny < 5:
        parser.error('nx must be positive, and ny must be at least 5 (upper five left edges)')
    if not math.isfinite(args.warm_end) or not 0 < args.warm_end <= DOMAIN_X_END:
        parser.error('warm-end must be finite and inside 0 < X <= 0.5 (nondimensional x)')
    if not (args.render_only or args.report_only):
        spacing = DOMAIN_X_END / args.nx
        warm_vertex = round(args.warm_end / spacing)
        if warm_vertex < 1 or abs(args.warm_end - warm_vertex * spacing) > BOUNDARY_TOLERANCE:
            parser.error(f'warm-end={args.warm_end:g} must coincide with a mesh vertex: '
                         f'choose X = k * (0.5/{args.nx}) for an integer k from 1 to {args.nx}, or change nx')
        outlet_vertex = round(MELT_OUTLET_END / spacing)
        if outlet_vertex < 1 or abs(MELT_OUTLET_END - outlet_vertex * spacing) > BOUNDARY_TOLERANCE:
            parser.error('The fixed melt outlet at x=0.05 must coincide with a mesh vertex; nx must be a multiple of 10')
    for name in ('jobs', 'fps', 'seconds', 'end_time', 'preheat_end_time', 'maximum_steps', 'preheat_maximum_steps'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            option = {'end_time': 'final-time', 'preheat_end_time': 'preheat-final-time'}.get(name, name.replace('_', '-'))
            parser.error(f'{option} must be finite and positive')
    size = f'{args.nx}x{args.ny}'
    args.preheat = (args.preheat or ROOT/f'src/couple/example/formalpreheat/output/evolved_{size}').resolve()
    args.output = (args.output or ROOT/f'src/couple/example/formalevolve/output/{size}').resolve()
    args.build_dir = args.build_dir.resolve()
    if args.preheat == args.output:
        parser.error('Preheat and evolution output directories must differ')
    return args


def configurations(args):
    """Derive both stage inputs from the existing physical case definitions."""
    import yaml
    preheat = yaml.safe_load((ROOT/'src/couple/example/formalpreheat/input.yaml').read_text())
    evolve = yaml.safe_load((ROOT/'src/couple/example/formalevolve/input.yaml').read_text())
    for config, stage, output, end, maximum in (
        (preheat, 'preheat', args.preheat, args.preheat_end_time, args.preheat_maximum_steps),
        (evolve, 'evolve', args.output, args.end_time, args.maximum_steps),
    ):
        config['mesh']['cells'] = [args.nx, args.ny]
        config['boundary_regions']['definitions']['left_upper']['selector']['start'] = args.ny - 5
        config['boundary_regions']['definitions']['warm_top']['selector']['interval'] = [0.0, args.warm_end]
        top_flow = next(segment for segment in config['flow']['boundary_conditions']['stokes']['segments']
                        if segment['region'] == 'top')
        top_flow['x']['value']['parameters']['cutoff_x'] = args.warm_end
        config['simulation']['name'] = f'formal_{stage}_{args.nx}x{args.ny}'
        config['output']['directory'] = str(output)
        config['time'].update(start=0.0, end=end, maximum_steps=maximum)
    # Retain the legacy cap of 10; reduce it automatically if a finer mesh requires it.
    preheat['time']['step'].update(control='cfl', minimum=1e-7)
    return preheat, evolve


def commands(args):
    """Ordered external stages. The preheat handoff is checked between the solvers."""
    result = []
    if not (args.render_only or args.report_only):
        targets = ['mantle_couple_preheat', 'mantle_couple_evolve'] if args.run_preheat else ['mantle_couple_evolve']
        result = [
            ('configure', ['cmake', '-S', str(ROOT), '-B', str(args.build_dir), '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=OFF',
                           '-DMANTLE_BUILD_DRIVER=OFF', f'-DHDF5_ROOT={args.hdf5_root.resolve()}']),
            ('build', ['cmake', '--build', str(args.build_dir), '--target', *targets, '--parallel', str(args.jobs)]),
        ]
        if args.run_preheat:
            result.append(('preheat', [str(args.build_dir/'src/couple/mantle_couple_preheat'), '-input', str(args.preheat/'preheat_input.yaml')]))
        result.append(('evolve', [str(args.build_dir/'src/couple/mantle_couple_evolve'), '-input', str(args.output/'evolve_input.yaml'),
                                   '-preheat', str(args.preheat), '-couple_movie_snapshots']))
        if args.run_preheat:
            result.append(('preheat_plots', [sys.executable, str(ROOT/'src/couple/plot_initialization.py'), '--input-dir', str(args.preheat)]))
    result.append(('conditions', [sys.executable, str(ROOT/'src/couple/conditions_report.py'), '--input-dir', str(args.output)]))
    if not args.report_only:
        result.append(('movies', [sys.executable, str(ROOT/'src/couple/animate_evolution.py'), '--input-dir', str(args.output),
                                  '--seconds', str(args.seconds), '--fps', str(args.fps)]))
    return result


def check_preheat(args, expected_end=None):
    import numpy as np
    manifest = json.loads((args.preheat/'transport_state.json').read_text())
    if manifest['status'] != 'complete' or [manifest['mesh']['nx'], manifest['mesh']['ny']] != [args.nx, args.ny]:
        raise ValueError('Completed preheat grid does not match the requested mesh')
    if not math.isfinite(manifest['time']) or manifest['time'] <= 0 or manifest['accepted_steps'] < 1:
        raise ValueError('Preheat must contain evolved H/C, not just initialization')
    if expected_end is not None and not math.isclose(manifest['time'], expected_end, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Preheat did not reach the requested end time')
    for field in ('H', 'C'):
        data = np.loadtxt(args.preheat/f'cell{field}1.dat', ndmin=2)
        if data.shape != (args.ny, args.nx) or not np.isfinite(data).all():
            raise ValueError(f'Invalid preheat {field} matrix: expected {args.ny} rows of {args.nx} values')


def check_dependencies(report_only=False):
    from conditions_report import renderer_python
    for module in (('numpy', 'yaml') if report_only else ('numpy', 'matplotlib', 'scipy', 'yaml')):
        if importlib.util.find_spec(module) is None:
            raise ValueError(f'Missing Python dependency: {module}')
    renderer_python()
    if report_only:
        return
    if not shutil.which('ffmpeg'):
        raise ValueError('ffmpeg is required for MP4 output')
    encoders = subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders'], text=True, stderr=subprocess.STDOUT)
    if 'libx264' not in encoders:
        raise ValueError('ffmpeg needs the libx264 encoder')


def run_command(name, command, output):
    print(f'\n[{name}] {shlex.join(command)}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output/f'run_{name}.log').open('w') as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def main(argv=None):
    args = parse_args(argv)
    stages = commands(args)
    postprocess = args.render_only or args.report_only
    action = 'Report saved conditions only' if args.report_only else ('Render existing evolution only' if args.render_only else ('Fresh preheat → full evolution' if args.run_preheat else 'Existing preheat → full evolution'))
    print(f'{action}\nMesh: {args.nx}x{args.ny}\nPreheat H/C: {args.preheat}\nEvolution output: {args.output}', flush=True)
    if not postprocess:
        print(f'Warm surface in generated input: x=0 to {args.warm_end:g} (nondimensional); solid-velocity cutoff matches', flush=True)
        print(f'Preheat end time: {args.preheat_end_time:g}' if args.run_preheat else 'Using completed preheat H/C', flush=True)
        print(f'Evolution end time after preheat: {args.end_time:g}', flush=True)
    if args.dry_run:
        if not postprocess:
            print(f'Prepare matching {args.nx}x{args.ny} inputs; upper-left boundary starts at edge {args.ny-5}.')
            if args.run_preheat:
                print(f'Write {args.preheat/"preheat_input.yaml"}; preheat uses CFL control with maximum step 10.')
            print(f'Write {args.output/"evolve_input.yaml"}.')
        for name, command in stages:
            print(f'[{name}] {shlex.join(command)}')
            if name == 'preheat':
                print('Validate completed preheat time, mesh and H/C matrices before starting evolution.')
        print(f'Conditions PDF: {args.output/"visualization/conditions.pdf"}')
        if not args.report_only:
            print(f'Combined video: {args.output/"visualization/combined_evolution.mp4"}')
        return
    check_dependencies(args.report_only)
    if not postprocess:
        import yaml
        if not args.run_preheat:
            check_preheat(args)
        preheat, evolve = configurations(args)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'evolve_input.yaml').write_text(yaml.safe_dump(evolve, sort_keys=False))
        if args.run_preheat:
            args.preheat.mkdir(parents=True, exist_ok=True)
            (args.preheat/'preheat_input.yaml').write_text(yaml.safe_dump(preheat, sort_keys=False))
    for name, command in stages:
        run_command(name, command, args.output)
        if name == 'evolve':
            # Archive only after a successful handoff, preserving older report
            # provenance if a new run fails before replacing its outputs.
            shutil.copyfile(args.preheat/'input_used.yaml', args.output/'preheat_input_used.yaml')
        if name == 'preheat':
            check_preheat(args, expected_end=args.preheat_end_time)
    print(f'\nConditions PDF: {args.output/"visualization/conditions.pdf"}')
    if not args.report_only:
        print(f'Open {args.output/"visualization/index.html"}')
        print(f'Combined video: {args.output/"visualization/combined_evolution.mp4"}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f'Run stopped: {error}', file=sys.stderr)
        raise SystemExit(1)
