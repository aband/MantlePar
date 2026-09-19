"""Check the two-stage runner and failure gates without invoking either solver."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_evolution as runner


class RunScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mantle-run-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.argv = ['--run-preheat', '--mesh', '40', '60', '--preheat-end-time', '25', '--end-time', '2',
                     '--preheat', str(self.root/'preheat output'), '--output', str(self.root/'evolution output')]

    def write_preheat(self, args, time=25):
        args.preheat.mkdir(parents=True, exist_ok=True)
        (args.preheat/'transport_state.json').write_text(json.dumps({
            'status':'complete', 'mesh':{'nx':args.nx,'ny':args.ny}, 'time':time, 'accepted_steps':3}))
        for field, value in (('H', 3.35), ('C', .03)):
            np.savetxt(args.preheat/f'cell{field}1.dat', np.full((args.ny,args.nx), value))
        preheat, _ = runner.configurations(args)
        (args.preheat/'input_used.yaml').write_text(yaml.safe_dump(preheat))

    def test_complete_pipeline(self):
        argv = self.argv + ['--warm-end', '0.05']
        args = runner.parse_args(argv)
        seen = []
        def command(name, command, output):
            seen.append(name)
            if name == 'preheat':
                preheat = yaml.safe_load(Path(command[-1]).read_text())
                assert preheat['mesh']['cells'] == [40,60]
                assert preheat['porosity']['prescribed_function']['parameters']['value'] == 0
                assert preheat['time']['end'] == 25
                assert preheat['time']['step']['control'] == 'cfl'
                assert preheat['time']['step']['maximum'] == 10
                assert preheat['boundary_regions']['definitions']['warm_top']['selector']['interval'] == [0.0,0.05]
                self.write_preheat(args)
            elif name == 'evolve':
                assert seen[-2] == 'preheat'
                source = Path(command[command.index('-preheat')+1])
                assert source == args.preheat
                assert (source/'cellH1.dat').is_file() and (source/'cellC1.dat').is_file()
                evolve = yaml.safe_load(Path(command[command.index('-input')+1]).read_text())
                assert evolve['mesh']['cells'] == [40,60]
                assert evolve['time']['start'] == 0 and evolve['time']['end'] == 2
                assert evolve['porosity']['source'] == 'gauss_point_data'
                assert evolve['boundary_regions']['definitions']['warm_top']['selector']['interval'] == [0.0,0.05]
                for field in ('enthalpy','composition'):
                    warm = next(segment for segment in evolve['transport']['boundary_conditions'][field]['segments'] if segment['region']=='warm_top')
                    assert warm['type'] == 'outflow' and 'value' not in warm
                assert '-couple_movie_snapshots' in command
        with patch.object(runner, 'check_dependencies'), patch.object(runner, 'run_command', side_effect=command), contextlib.redirect_stdout(io.StringIO()):
            runner.main(argv)
        self.assertEqual(seen, ['configure','build','preheat','evolve','preheat_plots','conditions','movies'])
        self.assertEqual((args.output/'preheat_input_used.yaml').read_text(), (args.preheat/'input_used.yaml').read_text())

    def test_preheat_failure_stops_before_evolution_even_with_old_output(self):
        args = runner.parse_args(self.argv)
        self.write_preheat(args)
        seen = []
        def fail(name, command, output):
            seen.append(name)
            if name == 'preheat':
                raise subprocess.CalledProcessError(7, command)
        with patch.object(runner, 'check_dependencies'), patch.object(runner, 'run_command', side_effect=fail), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.main(self.argv)
        self.assertEqual(seen, ['configure','build','preheat'])

    def test_handoff_validation_stops_before_evolution(self):
        args = runner.parse_args(self.argv)
        for kind in ('wrong_time', 'wrong_mesh', 'truncated_H', 'initial_only', 'incomplete'):
            with self.subTest(kind=kind):
                seen = []
                def broken(name, command, output):
                    seen.append(name)
                    if name != 'preheat':
                        return
                    self.write_preheat(args)
                    path = args.preheat/'transport_state.json'
                    meta = json.loads(path.read_text())
                    if kind == 'wrong_time': meta['time'] = 20
                    if kind == 'wrong_mesh': meta['mesh']['ny'] = 40
                    if kind == 'truncated_H': (args.preheat/'cellH1.dat').write_text('3.35\n')
                    if kind == 'initial_only': meta.update(time=0, accepted_steps=0)
                    if kind == 'incomplete': meta['status'] = 'incomplete'
                    path.write_text(json.dumps(meta))
                with patch.object(runner, 'check_dependencies'), patch.object(runner, 'run_command', side_effect=broken), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError):
                        runner.main(self.argv)
                self.assertNotIn('evolve', seen)

    def test_geometry_and_template_compatibility(self):
        original = yaml.safe_load((runner.ROOT/'src/couple/example/formalevolve/input.yaml').read_text())
        for nx,ny in ((20,20),(40,40),(80,80),(40,60),(20,5)):
            args = runner.parse_args(['--mesh',str(nx),str(ny)])
            preheat,evolve = runner.configurations(args)
            for config in (preheat,evolve):
                self.assertEqual(config['mesh']['cells'], [nx,ny])
                definitions = config['boundary_regions']['definitions']
                top_flow = next(segment for segment in config['flow']['boundary_conditions']['stokes']['segments'] if segment['region']=='top')
                self.assertEqual(top_flow['x']['value']['parameters']['cutoff_x'], definitions['warm_top']['selector']['interval'][1])
                self.assertEqual(definitions['left_upper']['selector'], {'mode':'boundary_cells','count':5,'start':ny-5})
                for region in ('warm_top','melt_outlet'):
                    if region in definitions:
                        for x in definitions[region]['selector']['interval']:
                            self.assertAlmostEqual(x*nx/.5, round(x*nx/.5))
            for key in ('phase','flow','porosity','transport','quadrature'):
                self.assertEqual(evolve[key], original[key])

    def test_render_only_skips_both_stages(self):
        seen = []
        with patch.object(runner, 'check_dependencies'), patch.object(runner, 'run_command', side_effect=lambda name,*_: seen.append(name)), contextlib.redirect_stdout(io.StringIO()):
            runner.main(self.argv+['--render-only'])
        self.assertEqual(seen, ['conditions','movies'])
        self.assertFalse((self.root/'preheat output').exists())

    def test_report_only_skips_simulations_and_movies(self):
        seen = []
        with patch.object(runner, 'check_dependencies'), patch.object(runner, 'run_command', side_effect=lambda name,*_: seen.append(name)), contextlib.redirect_stdout(io.StringIO()):
            runner.main(self.argv+['--report-only'])
        self.assertEqual(seen, ['conditions'])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_configurable_warm_endpoint_and_mesh_alignment(self):
        for nx, endpoint in ((10,.05), (30,.05), (60,.075), (30,.3), (20,.5)):
            with self.subTest(nx=nx, endpoint=endpoint):
                args = runner.parse_args(['--grid',str(nx),'--warm-end',str(endpoint)])
                preheat,evolve = runner.configurations(args)
                for config in (preheat,evolve):
                    self.assertEqual(config['boundary_regions']['definitions']['warm_top']['selector']['interval'], [0.0,endpoint])
                    warm = next(segment for segment in config['transport']['boundary_conditions']['temperature']['segments'] if segment['region']=='warm_top')
                    self.assertEqual(warm['value']['parameters']['value'], 3.35)
                    top_flow = next(segment for segment in config['flow']['boundary_conditions']['stokes']['segments'] if segment['region']=='top')
                    self.assertEqual(top_flow['x']['value']['parameters']['cutoff_x'], endpoint)
                    self.assertEqual(top_flow['x']['value']['parameters']['amplitude'], 2.029426686961948e-5)
                    self.assertEqual(top_flow['y']['value']['parameters']['value'], 0.0)
                self.assertEqual(evolve['boundary_regions']['definitions']['melt_outlet']['selector']['interval'], [0.0,.05])
                for field in ('enthalpy','composition'):
                    warm = next(segment for segment in evolve['transport']['boundary_conditions'][field]['segments'] if segment['region']=='warm_top')
                    self.assertEqual(warm, {'region':'warm_top','priority':20,'type':'outflow'})
                    self.assertFalse(any(segment['region']=='warm_top' for segment in preheat['transport']['boundary_conditions'][field].get('segments', [])))
        # Rendering an existing non-default mesh uses the recorded field data;
        # it does not require specifying the original thermal boundary again.
        runner.parse_args(['--grid','30','--render-only'])
        for flags, message in (
            (['--warm-end','nan'], 'finite'),
            (['--warm-end','inf'], 'finite'),
            (['--warm-end','0'], '0 < X <= 0.5'),
            (['--warm-end','-0.025'], '0 < X <= 0.5'),
            (['--warm-end','0.6'], '0 < X <= 0.5'),
            (['--warm-end','1e-13'], 'mesh vertex'),
            (['--grid','30'], 'mesh vertex'),
            (['--grid','20','--warm-end','0.04'], 'mesh vertex'),
            (['--grid','12','--warm-end','0.125'], 'fixed melt outlet'),
        ):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as caught:
                    runner.parse_args(flags)
                self.assertEqual(caught.exception.code, 2)
                self.assertIn(message, error.getvalue())

    def test_shell_entry_points_and_dry_run_no_writes(self):
        for script, fresh in (('run.sh',True), ('evolve.sh',False)):
            result = subprocess.run([str(runner.ROOT/script), *self.argv[1:], '--dry-run'], cwd=self.root, text=True, capture_output=True, check=True)
            self.assertEqual('[preheat]' in result.stdout, fresh)
            self.assertIn('40x60', result.stdout)
            self.assertIn('combined_evolution.mp4', result.stdout)
        self.assertEqual(list(self.root.iterdir()), [])
        result = subprocess.run([str(runner.ROOT/'run.sh'), '--grid','30','--warm-end','0.05','--dry-run'], cwd=self.root, text=True, capture_output=True, check=True)
        self.assertIn('30x30', result.stdout)
        self.assertIn('x=0 to 0.05', result.stdout)
        for flags in (['--grid','30'], ['--mesh','20','4'], ['--end-time','nan'], ['--preheat-end-time','0']):
            result = subprocess.run([str(runner.ROOT/'run.sh'), *flags, '--dry-run'], cwd=self.root, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
