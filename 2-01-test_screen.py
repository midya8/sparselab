import copy
from fractions import Fraction as F
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from sparselab import screen, screen_demo, enumerate_optima, demo
ROOT=Path(__file__).resolve().parents[1]


def fsum(values):return sum(values,F(0))


class ScreenTests(unittest.TestCase):
    def test_exact_equations_every_control(self):
        for row in screen_demo()['controls']:
            s=row['screen'];x=[[F(v)for v in r]for r in s['X']];y=[F(v)for v in s['y']]
            beta=[F(v)for v in s['coefficients']];u=[F(v)for v in s['dual_vector']];lam=F(s['penalty']);n=s['n']
            residual=[yi-fsum(a*b for a,b in zip(xx,beta))for xx,yi in zip(x,y)]
            primal=fsum(v*v for v in residual)/(2*n)+lam*fsum(abs(v)for v in beta)
            dual=fsum(yi*ui for yi,ui in zip(y,u))-n*fsum(ui*ui for ui in u)/2
            self.assertEqual(F(s['primal']),primal);self.assertEqual(F(s['dual']),dual)
            self.assertEqual(F(s['gap']),primal-dual)
            self.assertEqual(F(s['radius_squared']),2*(primal-dual)/n)
            self.assertEqual([F(v)for v in s['residual']],residual)
            for j,feature in enumerate(s['features']):
                corr=fsum(xx[j]*ui for xx,ui in zip(x,u));norm=fsum(xx[j]**2 for xx in x)
                margin=lam-abs(corr);ball=norm*F(s['radius_squared'])
                self.assertLessEqual(abs(corr),lam)
                self.assertEqual(F(feature['center_correlation']),corr)
                self.assertEqual(F(feature['margin']),margin)
                self.assertEqual(F(feature['column_norm_squared']),norm)
                self.assertEqual(F(feature['margin_squared']),margin**2)
                self.assertEqual(F(feature['ball_term']),ball)
                self.assertEqual(F(feature['strict_slack']),margin**2-ball)
                self.assertEqual(feature['safe_zero'],margin>0 and ball<margin**2)

    def test_expected_positive_and_negative_controls(self):
        r=screen_demo()['controls']
        self.assertEqual([s['screen']['safe_zero']for s in r],[[],[2],[2],[],[],[1]])
        self.assertEqual(r[1]['screen']['gap'],'5/1024')
        self.assertEqual(r[1]['screen']['coefficients'][2],'1/32')
        self.assertEqual(r[1]['screen']['features'][2]['strict_slack'],'15/1024')
        for i in [2,3,4]:self.assertEqual(r[i]['screen']['gap'],'0')
        self.assertEqual(r[3]['screen']['coefficients'][2],'0')
        self.assertEqual(r[3]['screen']['features'][2]['margin'],'0')
        self.assertEqual(r[4]['exact']['witness_count'],2)
        self.assertEqual(r[4]['screen']['not_certified'],[0,1,2])

    def test_strict_sphere_boundary_not_less_equal(self):
        s=screen([[1]],[.75],[.25],1)
        self.assertEqual(s['gap'],'1/8');self.assertEqual(s['radius_squared'],'1/4')
        self.assertEqual(s['features'][0]['margin_squared'],'1/4')
        self.assertEqual(s['features'][0]['ball_term'],'1/4')
        self.assertEqual(s['features'][0]['strict_slack'],'0')
        self.assertEqual(s['safe_zero'],[])
        # A slightly tighter real rational witness certifies the same true zero.
        self.assertEqual(screen([[1]],[.75],[.125],1)['safe_zero'],[0])

    def test_balls_contain_independent_exact_dual_optimum(self):
        for row in screen_demo()['controls']:
            s=row['screen'];n=s['n'];center=[F(v)for v in s['dual_vector']]
            for optimum in row['exact']['witnesses']:
                ustar=[(F(yi)-F(fi))/n for yi,fi in zip(s['y'],optimum['prediction'])]
                distance=fsum((a-b)**2for a,b in zip(center,ustar))
                self.assertLessEqual(distance,F(s['radius_squared']))
                self.assertLessEqual(F(s['dual']),F(optimum['primal']))
                self.assertGreaterEqual(F(s['primal']),F(optimum['primal']))

    def test_seeded_independent_oracle_soundness_and_containment(self):
        rng=random.Random(431)
        for _ in range(24):
            x=[[rng.randint(-3,3)for _ in range(3)]for _ in range(5)]
            y=[rng.randint(-4,4)for _ in range(5)];beta=[rng.choice([0,.125,-.25,.5])for _ in range(3)]
            lam=rng.choice([.125,.25,1,2]);s=screen(x,y,beta,lam);oracle=enumerate_optima(x,y,lam)
            for w in oracle['witnesses']:
                self.assertTrue(all(F(w['coefficients'][j])==0 for j in s['safe_zero']))
                ustar=[(F(yi)-F(fi))/5for yi,fi in zip(y,w['prediction'])]
                self.assertLessEqual(fsum((F(c)-v)**2for c,v in zip(s['dual_vector'],ustar)),F(s['radius_squared']))

    def test_zero_columns_and_zero_residual_scale(self):
        s=screen([[0,0]],[0],[7,-8],.25)
        self.assertEqual(s['safe_zero'],[0,1]);self.assertEqual(s['dual_scale'],'1')
        self.assertEqual(s['dual_vector'],['0']);self.assertGreater(F(s['gap']),0)
        self.assertTrue(all(f['ball_term']=='0'for f in s['features']))
        s=screen([[1]],[0],[0],.25)
        self.assertEqual(s['gap'],'0');self.assertEqual(s['safe_zero'],[0])

    def test_binary_float_semantics_not_decimal_idealization(self):
        s=screen([[1]],[.3],[.1],.1)
        self.assertEqual(F(s['y'][0]),F(.3));self.assertNotEqual(F(s['y'][0]),F(3,10))
        self.assertEqual(F(s['coefficients'][0]),F(.1))
        self.assertEqual(F(s['penalty']),F(.1))
        self.assertEqual(F(s['residual'][0]),F(.3)-F(.1))

    def test_row_column_permutations_and_signs(self):
        r=screen_demo()['controls'][1]['screen']
        x=[[float(F(v))for v in row]for row in r['X']];y=[float(F(v))for v in r['y']];beta=[float(F(v))for v in r['coefficients']]
        a=screen(list(reversed(x)),list(reversed(y)),beta,.25)
        self.assertEqual(a['features'],r['features']);self.assertEqual(a['gap'],r['gap'])
        a=screen([list(reversed(row))for row in x],y,list(reversed(beta)),.25)
        self.assertEqual(a['safe_zero'],[0]);self.assertEqual(a['gap'],r['gap'])
        a=screen(x,[-v for v in y],[-v for v in beta],.25)
        self.assertEqual(a['safe_zero'],r['safe_zero']);self.assertEqual(a['gap'],r['gap'])

    def test_row_replication_mean_normalization(self):
        r=screen_demo()['controls'][1]['screen']
        x=[[float(F(v))for v in row]for row in r['X']];y=[float(F(v))for v in r['y']];beta=[float(F(v))for v in r['coefficients']]
        s=screen(x+x,y+y,beta,.25)
        self.assertEqual(s['gap'],r['gap']);self.assertEqual(s['primal'],r['primal']);self.assertEqual(s['dual'],r['dual'])
        self.assertEqual(2*F(s['radius_squared']),F(r['radius_squared']))
        self.assertEqual(s['features'],[dict(f,column_norm_squared=str(2*F(f['column_norm_squared'])))for f in r['features']])
        self.assertEqual(s['safe_zero'],[2])

    def test_column_and_response_scale_equivariance(self):
        r=screen_demo()['controls'][1]['screen']
        x=[[float(F(v))for v in row]for row in r['X']];y=[float(F(v))for v in r['y']];beta=[float(F(v))for v in r['coefficients']]
        a=screen([[2*v for v in row]for row in x],y,[v/2 for v in beta],.5)
        self.assertEqual(a['gap'],r['gap']);self.assertEqual(a['radius_squared'],r['radius_squared']);self.assertEqual(a['safe_zero'],r['safe_zero'])
        for aa,rr in zip(a['features'],r['features']):self.assertEqual(F(aa['strict_slack']),4*F(rr['strict_slack']))
        a=screen(x,[2*v for v in y],[2*v for v in beta],.5)
        self.assertEqual(F(a['gap']),4*F(r['gap']));self.assertEqual(a['safe_zero'],[2])

    def test_no_mutation_rng_and_general_fit_size(self):
        x=[[1,0],[1,0]];y=[2,2];beta=[0,3];before=copy.deepcopy((x,y,beta));state=random.getstate()
        screen(x,y,beta,.25);self.assertEqual((x,y,beta),before);self.assertEqual(state,random.getstate())
        s=screen([[0]*12 for _ in range(128)],[0]*128,[0]*12,.25)
        self.assertEqual(s['safe_zero'],list(range(12)))

    def test_bad_input_and_no_silent_coef_conversion(self):
        for beta in [[],[True],[float('nan')],[1e7],['0']]:
            with self.assertRaises(ValueError):screen([[1]],[1],beta,.25)
        for args in [([],[],[],.25),([[1e-6]],[1],[0],.25),([[1]],[1],[0],0),([[1]],[True],[0],.25)]:
            with self.assertRaises(ValueError):screen(*args)

    def test_all_cli_paths_committed_reports_and_old_unchanged(self):
        self.assertEqual(screen_demo(),json.loads((ROOT/'examples/screen-demo.json').read_text()))
        commands=[('demo',None,demo()),('screen-demo',None,screen_demo()),('screen','examples/screen-input.json',screen_demo()['controls'][1]['screen'])]
        for command,filename,expected in commands:
            args=[sys.executable,'-m','sparselab',command]+([str(ROOT/filename)]if filename else [])
            self.assertEqual(json.loads(subprocess.check_output(args,cwd=ROOT)),expected)
        self.assertEqual(demo(),json.loads((ROOT/'examples/demo.json').read_text()))

    def test_strict_screen_cli(self):
        original=json.loads((ROOT/'examples/screen-input.json').read_text())
        bad=[dict(original,penalties=[1]),dict(original,schema_version=True),dict(original,coefficients=[True,0,0])]
        missing=dict(original);del missing['coefficients'];bad.append(missing)
        bodies=[json.dumps(v).encode()for v in bad]+[b'{"schema_version":1,"schema_version":1}',b'NaN',b'\xff',b' '*100001]
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)/'input.json'
            for body in bodies:
                p.write_bytes(body)
                r=subprocess.run([sys.executable,'-m','sparselab','screen',str(p)],capture_output=True,cwd=ROOT)
                self.assertEqual(r.returncode,2);self.assertFalse(r.stdout)
