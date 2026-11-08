import base64
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zlib
from sparselab import demo
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('render_figure',ROOT/'examples/render_figure.py')
FIG=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(FIG)


class FigureTests(unittest.TestCase):
    def test_visible_coefficients_and_all_geometry(self):
        report=demo();root=ET.fromstring(FIG.render(report));byid={e.get('id'):e for e in root.iter()if e.get('id')}
        for j in range(3):
            expected=[]
            for k,r in enumerate(report['orthogonal']['path']['results']):
                xx=124+174*k;yy=330-74*r['coefficients'][j]
                p=byid[f'point-{j}-{k}'];self.assertEqual(float(p.get('cx')),xx);self.assertEqual(float(p.get('cy')),yy)
                expected.append((xx,yy))
            points=[tuple(map(float,p.split(',')))for p in byid[f'coefficient-{j}'].get('points').split()]
            self.assertEqual(points,expected)
        for i,key in enumerate(['forward','reverse','midpoint']):
            r=report['duplicate'][key]
            for j,(xx,v)in enumerate(zip([345,465,585,710,830],r['coefficients']+[r['primal'],r['max_kkt']])):
                e=byid[f'duplicate-{i}-{j}'];self.assertEqual(float(e.get('data-value')),v);self.assertEqual(e.text,f'{v:.6g}')
                self.assertEqual(float(e.get('x')),xx);self.assertEqual(float(e.get('y')),582+i*38)
        for i,key in enumerate(['limited','converged']):
            r=report['correlated'][key]
            for j,(xx,v)in enumerate(zip([345,465,650,830],[r['sweeps'],r['primal'],r['gap'],r['max_kkt']])):
                e=byid[f'convergence-{i}-{j}'];self.assertEqual(float(e.get('data-value')),v);self.assertEqual(e.text,f'{v:.6g}')
                self.assertEqual(float(e.get('x')),xx);self.assertEqual(float(e.get('y')),852+i*35)

    def test_complete_report_and_hidden_tamper(self):
        r=demo();svg=FIG.render(r);root=ET.fromstring(svg)
        metadata=next(e for e in root.iter()if e.get('id')=='full-report')
        decoded=json.loads(zlib.decompress(base64.b64decode(metadata.text)))
        self.assertEqual(decoded,r)
        altered=copy.deepcopy(r);altered['correlated']['converged']['history'][3]['gap']+=.001
        self.assertNotEqual(FIG.render(altered),svg)

    def test_committed_figure_byte_exact(self):
        self.assertEqual((ROOT/'assets/sparsity-audit.svg').read_text(),FIG.render(demo()))
