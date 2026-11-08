import base64
import copy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zlib
from sparselab import screen_demo
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('screen_figure',ROOT/'examples/render_screening.py')
FIG=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(FIG)


class ScreenFigureTests(unittest.TestCase):
    def test_all_visible_cells_and_squared_bound_bars(self):
        report=screen_demo();root=ET.fromstring(FIG.render(report));ids={e.get('id'):e for e in root.iter()if e.get('id')}
        for i,row in enumerate(report['controls']):
            s=row['screen']
            for j,(xx,v)in enumerate(zip([430,635,860],[s['gap'],s['radius_squared'],str(s['safe_zero'])])):
                e=ids[f'control-{i}-{j}'];self.assertEqual(e.text,v);self.assertEqual(float(e.get('x')),xx);self.assertEqual(float(e.get('y')),220+43*i)
        for i,f in enumerate(report['controls'][1]['screen']['features']):
            for j,(xx,v)in enumerate(zip([250,575,835],[f['ball_term'],f['margin_squared'],f['strict_slack']])):
                e=ids[f'near-{i}-{j}'];self.assertEqual(e.text,v);self.assertEqual(float(e.get('x')),xx);self.assertEqual(float(e.get('y')),652+37*i)
        f=report['controls'][1]['screen']['features'][2]
        for i,key in enumerate(['ball_term','margin_squared']):
            e=ids[f'bound-bar-{i}'];self.assertEqual(float(e.get('width')),float(Fraction(f[key]))*18000)
            self.assertEqual(e.get('x'),'250');self.assertEqual(float(e.get('y')),782+i*47);self.assertEqual(e.get('height'),'26')

    def test_full_exact_metadata_and_hidden_dual_tamper(self):
        report=screen_demo();svg=FIG.render(report);root=ET.fromstring(svg)
        metadata=next(e for e in root.iter()if e.get('id')=='full-screen-report')
        self.assertEqual(json.loads(zlib.decompress(base64.b64decode(metadata.text))),report)
        altered=copy.deepcopy(report);altered['controls'][0]['screen']['dual_vector'][2]='999/7'
        self.assertNotEqual(FIG.render(altered),svg)

    def test_committed_figure_byte_exact(self):
        self.assertEqual((ROOT/'assets/safe-screening.svg').read_text(),FIG.render(screen_demo()))
