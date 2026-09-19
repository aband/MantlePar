"""Verify report provenance and saved boundary settings without solving physics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from conditions_report import load_recorded, generate, scales, region_extent, digest
from run_evolution import configurations, parse_args


class ConditionsReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mantle-report-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/'preheat'; self.output=self.root/'evolution'
        self.source.mkdir(); self.output.mkdir()
        args=parse_args(['--mesh','30','40','--warm-end','0.05','--final-time','2',
                         '--preheat',str(self.source),'--output',str(self.output)])
        self.preheat,self.evolve=configurations(args)
        for path,c in [(self.source/'input_used.yaml',self.preheat),
                       (self.output/'preheat_input_used.yaml',self.preheat),
                       (self.output/'input_used.yaml',self.evolve)]:
            path.write_text(yaml.safe_dump(c,sort_keys=False))
        (self.output/'visualization.json').write_text(json.dumps(dict(status='complete',evolution_model='full phase coupling',nx=30,ny=40,time=2)))
        (self.output/'source_state.json').write_text(json.dumps(dict(directory=str(self.source),source_time=15000,evolution_start=0)))
        np.savetxt(self.output/'starting_cellH.dat',np.full((40,30),3.2))
        np.savetxt(self.output/'starting_cellC.dat',np.full((40,30),.04))

    def test_saved_conditions_and_handoff(self):
        # Reusing the source folder must not change this earlier run's report.
        (self.source/'input_used.yaml').write_text('unrelated: replacement\n')
        data=load_recorded(self.output)
        self.assertEqual(data['configurations'],[self.preheat,self.evolve])
        self.assertEqual(data['inputs'][0]['sha256'],digest(self.output/'preheat_input_used.yaml'))
        self.assertEqual([(v['minimum'],v['maximum'],v['count']) for v in data['imported']],[(3.2,3.2,1200),(.04,.04,1200)])
        self.assertIn('Archived',data['preheat_provenance'])
        s=scales(self.evolve)
        self.assertAlmostEqual(s['temperature_K'],520)
        self.assertAlmostEqual(s['velocity_m_s']*100*365*24*3600*2.029426686961948e-5,3.2)
        region=region_extent(self.evolve,self.evolve['boundary_regions']['definitions']['warm_top'])
        self.assertIn('[0, 0.05]',region)
        self.assertIn('15.811388',region)

    def test_old_run_provenance_is_explicit(self):
        (self.output/'preheat_input_used.yaml').unlink()
        data=load_recorded(self.output)
        self.assertIn('Older run',data['preheat_provenance'])
        self.assertEqual(data['inputs'][0]['path'],str(self.source/'input_used.yaml'))

    def test_incomplete_or_mismatched_output_rejected(self):
        path=self.output/'visualization.json'; meta=json.loads(path.read_text())
        for key,value in [('status','incomplete'),('nx',20),('time',1)]:
            path.write_text(json.dumps({**meta,key:value}))
            with self.assertRaises(ValueError): load_recorded(self.output)
        path.write_text(json.dumps(meta))
        (self.output/'starting_cellH.dat').write_text('3.2\n')
        with self.assertRaises(ValueError): load_recorded(self.output)

    def test_pdf_contains_conditions_and_html_link(self):
        data=load_recorded(self.output)
        path=generate(data,self.output/'visualization/conditions.pdf')
        text=subprocess.check_output(['pdftotext',str(path),'-'],text=True)
        for token in ('warm_top','melt_outlet','outflow','backflow is rejected','1742 K','3.2 cm/year',
                      'cutoff_x=0.05','pressure_potential','starting_cellH.dat','starting_cellC.dat',
                      'Archived with evolution output','Zero normal diffusive heat flux'):
            self.assertIn(token,text)
        self.assertGreaterEqual(text.count('\f'),8)
        self.assertEqual(json.loads(path.with_suffix('.json').read_text())['configurations'],data['configurations'])
        from types import SimpleNamespace
        from animate_evolution import write_notes
        fixture=SimpleNamespace(meta={'nx':30,'ny':40,'eutectic_reference_K':1480},
                                clock=np.array([[0,0,0],[1,2,401.1]]),final={})
        write_notes(fixture,path.parent,{'velocity':'velocity_evolution.mp4'})
        self.assertIn('href="conditions.pdf"',(path.parent/'index.html').read_text())
        self.assertIn('conditions.pdf',(path.parent/'report.md').read_text())


if __name__=='__main__': unittest.main()
