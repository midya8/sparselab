import base64
import copy
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zlib
from sparselab import branch_demo
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('branch_figure',ROOT/'examples/render_branches.py')
FIG=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(FIG)


class BranchFigureTests(unittest.TestCase):
    def test_all_twenty_cells_and_full_path_geometry(self):
        report=branch_demo();root=ET.fromstring(FIG.render(report));ids={e.get('id'):e for e in root.iter()if e.get('id')}
        rows=[['[0, 0]','[3, infinity)','0','0'],['[0, 1]','[4/3, 3]','0','3/5 - 1/5 * lambda'],
              ['[1, 1]','[1, 4/3]','4 - 3 * lambda','-1 + 1 * lambda'],
              ['[1, 0]','[1/3, 1]','2 - 1 * lambda','0'],['[1, -1]','[0, 1/3]','4 - 7 * lambda','-1 + 3 * lambda']]
        for i,row in enumerate(rows):
            for j,(xx,value)in enumerate(zip([58,300,550,810],row)):
                e=ids[f'branch-cell-{i}-{j}'];self.assertEqual(e.text,value);self.assertEqual(float(e.get('x')),xx);self.assertEqual(float(e.get('y')),769+39*i)
        points=[(F(3),[F(0),F(0)]),(F(4,3),[F(0),F(1,3)]),(F(1),[F(1),F(0)]),(F(1,3),[F(5,3),F(0)]),(F(0),[F(4),F(-1)])]
        for j in [0,1]:
            coords=[tuple(map(float,p.split(',')))for p in ids[f'coefficient-path-{j}'].get('points').split()]
            expected=[(float(110+(3-lam)*300),float(520-beta[j]*66))for lam,beta in points]
            self.assertEqual(coords,expected)
            for i,(xx,yy)in enumerate(expected):
                e=ids[f'vertex-{j}-{i}'];self.assertEqual(float(e.get('cx')),xx);self.assertEqual(float(e.get('cy')),yy);self.assertEqual(e.get('r'),'5')
        for i,(lam,beta)in enumerate(points):
            e=ids[f'knot-{i}'];self.assertEqual(e.text,str(lam));self.assertEqual(float(e.get('x')),float(110+(3-lam)*300));self.assertEqual(e.get('y'),'618')

    def test_full_metadata_and_hidden_singular_gram_tamper(self):
        report=branch_demo();svg=FIG.render(report);root=ET.fromstring(svg)
        metadata=next(e for e in root.iter()if e.get('id')=='full-branch-report')
        self.assertEqual(json.loads(zlib.decompress(base64.b64decode(metadata.text))),report)
        changed=copy.deepcopy(report);changed['singular']['gram'][0][0]='999/7';self.assertNotEqual(FIG.render(changed),svg)

    def test_committed_figure_byte_exact(self):
        self.assertEqual((ROOT/'assets/regularization-branches.svg').read_text(),FIG.render(branch_demo()))
