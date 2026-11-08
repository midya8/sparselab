import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from sparselab import fit, path, witness, enumerate_optima, demo
from sparselab.core import soft
from sparselab.oracle import _solve

ROOT = Path(__file__).resolve().parents[1]
X = [[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]
Y = [1.375,2.625,-2.875,-1.125]


class SparseTests(unittest.TestCase):
    def assertClose(self, a, b, tol=1e-9):
        self.assertLessEqual(abs(a-b), tol)

    def test_orthogonal_closed_form_every_penalty(self):
        for lam in [2,1,.5,.25,.125,.0625]:
            result = fit(X,Y,lam)
            expected = [max(2-lam,0),-max(.75-lam,0),max(.125-lam,0)]
            self.assertEqual(result['coefficients'], expected)
            self.assertTrue(result['converged'])
            self.assertEqual(result['sweeps'], 0 if lam >= 2 else 1)

    def test_scalar_nonunit_scale(self):
        for a in [-3, .5, 4]:
            for y in [-2,0,3]:
                for lam in [.125, .75, 3]:
                    expected=math.copysign(max(abs(a*y)-lam,0),a*y)/(a*a)
                    self.assertClose(fit([[a]],[y],lam)['coefficients'][0],expected)

    def test_full_primal_dual_equations(self):
        result=fit([[1,2],[2,-1],[-1,.5]],[1,-2,3],.125)
        n=3;beta=result['coefficients'];r=result['residual'];u=result['dual_vector']
        self.assertTrue(result['converged'])
        self.assertClose(result['primal'],sum(v*v for v in r)/(2*n)+.125*sum(abs(v) for v in beta))
        self.assertClose(result['dual'],sum(v*w for v,w in zip([1,-2,3],u))-n*sum(v*v for v in u)/2)
        self.assertLessEqual(result['dual_feasibility_max'],.125)
        self.assertClose(result['gap'],result['primal']-result['dual'])
        self.assertGreaterEqual(result['gap'],0)
        self.assertTrue(all(v>=0 for v in result['gap_terms']))
        for j,b in enumerate(beta):
            corr=sum(row[j]*v for row,v in zip([[1,2],[2,-1],[-1,.5]],r))/n
            self.assertClose(result['correlations'][j],corr)
            self.assertClose(corr, math.copysign(.125,b)) if b else self.assertLessEqual(abs(corr),.125+1e-10)

    def test_arbitrary_witness_weak_duality_and_gap(self):
        exact=enumerate_optima(X,Y,.25)
        optimum=float(Fraction(exact['witnesses'][0]['primal']))
        for beta in ([0,0,0],[2,-1,1],[100,-75,12],[-2,.5,-1]):
            w=witness(X,Y,beta,.25)
            self.assertGreaterEqual(w['primal']+1e-12,optimum)
            self.assertLessEqual(w['dual'],optimum+1e-12)
            self.assertGreaterEqual(w['gap']+1e-10,w['primal']-optimum)
            self.assertClose(w['gap'],w['direct_gap'],1e-9)
            self.assertLessEqual(w['dual_feasibility_max'],.25)

    def test_exact_oracle_rational_scalar(self):
        r=enumerate_optima([[2],[2]],[3,3],.25)
        self.assertEqual(r['witnesses'][0]['coefficients'],['23/16'])
        self.assertEqual(r['witnesses'][0]['primal'],'47/128')
        self.assertEqual(r['considered'],3)

    def test_fraction_gaussian_elimination_pivot_singular(self):
        F=Fraction
        self.assertEqual(_solve([[F(0),F(1)],[F(2),F(3)]],[F(1),F(5)]),[F(1),F(1)])
        self.assertIsNone(_solve([[F(1),F(1)],[F(2),F(2)]],[F(1),F(2)]))
        self.assertEqual(_solve([],[]),[])

    def test_exact_oracle_seeded_full_rank_independent(self):
        rng=random.Random(17)
        for _ in range(10):
            x=[[rng.randint(-4,4) for _ in range(3)] for _ in range(6)]
            y=[rng.randint(-3,3) for _ in range(6)]
            lam=rng.choice([.125,.25,.5,1])
            oracle=enumerate_optima(x,y,lam)
            result=fit(x,y,lam)
            self.assertTrue(result['converged'])
            self.assertTrue(oracle['objectives_agree'] and oracle['predictions_agree'])
            expected=oracle['witnesses'][0]
            self.assertClose(result['primal'],float(Fraction(expected['primal'])),1e-9)
            for a,b in zip(result['prediction'],expected['prediction']):self.assertClose(a,float(Fraction(b)),1e-8)
            self.assertEqual(oracle['considered'],27)

    def test_duplicate_order_and_convex_mixture(self):
        r=demo()['duplicate'];forward=r['forward'];reverse=r['reverse'];exact=r['exact']
        self.assertEqual(forward['coefficients'],[1.75,0,-.5])
        self.assertEqual(reverse['coefficients'],[0,1.75,-.5])
        self.assertEqual(exact['witness_count'],2)
        self.assertEqual(exact['singular_sign_patterns'],12)
        self.assertTrue(exact['predictions_agree'] and exact['objectives_agree'])
        for t in [0,.125,.25,.5,.875,1]:
            w=witness(r['X'],r['y'],[1.75*t,1.75*(1-t),-.5],.25)
            self.assertEqual(w['prediction'],forward['prediction'])
            self.assertEqual(w['primal'],.625)
            self.assertLess(w['gap'],1e-12)
            self.assertEqual(w['max_kkt'],0)
        self.assertEqual([w['primal'] for w in exact['witnesses']],['5/8','5/8'])

    def test_correlated_honest_sweep_limit(self):
        r=demo()['correlated'];short=r['limited'];long=r['converged']
        self.assertFalse(short['converged']);self.assertEqual(short['status'],'sweep_limit')
        self.assertEqual(short['sweeps'],1)
        self.assertGreater(short['gap'],.3);self.assertGreater(short['max_kkt'],.17)
        self.assertEqual(long['sweeps'],18)
        self.assertTrue(long['converged'])
        self.assertClose(long['coefficients'][1],33/17)
        self.assertClose(long['primal'],19/68)
        self.assertEqual(r['exact']['witnesses'][0]['primal'],'19/68')

    def test_monotone_sweep_objective(self):
        for r in [demo()['correlated']['converged'],fit(X,Y,.25,initial=[5,-8,4])]:
            for a,b in zip(r['history'],r['history'][1:]):self.assertLessEqual(b['primal'],a['primal']+1e-12)
            self.assertEqual(len(r['history']),r['sweeps']+1)
            self.assertEqual(r['history'][-1]['gap'],r['gap'])

    def test_warm_path_matches_cold_and_lambda_max(self):
        result=path(X,Y,[2,1,.5,.25,.125,.0625])
        self.assertEqual(result['lambda_max'],2)
        previous=[0.0]*3
        for row in result['results']:
            self.assertEqual(row['initial'],previous)
            cold=fit(X,Y,row['penalty'])
            self.assertEqual(row['coefficients'],cold['coefficients'])
            previous=row['coefficients']
        self.assertTrue(result['warm_started'] and result['all_converged'])

    def test_lambda_max_threshold_exact_zero(self):
        self.assertEqual(fit(X,Y,2)['coefficients'],[0,0,0])
        self.assertNotEqual(fit(X,Y,2-1e-6)['coefficients'],[0,0,0])
        r=fit(X,[0]*4,.25)
        self.assertEqual(r['dual_vector'],[0]*4);self.assertEqual(r['gap'],0)
        self.assertEqual(r['sweeps'],0)

    def test_zero_column_and_all_zero(self):
        r=fit([[1,0],[2,0]],[2,4],.25,initial=[0,3])
        self.assertEqual(r['coefficients'][1],0)
        q=fit([[0,0]],[2],.25,initial=[1,2])
        self.assertEqual(q['coefficients'],[0,0]);self.assertTrue(q['converged'])
        self.assertEqual(enumerate_optima([[0,0]],[2],.25)['witness_count'],1)

    def test_permutations_and_response_sign(self):
        x=[[1,2],[2,-1],[-1,.5]];y=[1,-2,3]
        baseline=fit(x,y,.25)
        rr=fit(list(reversed(x)),list(reversed(y)),.25)
        for a,b in zip(rr['coefficients'],baseline['coefficients']):self.assertClose(a,b)
        rr=fit([list(reversed(row))for row in x],y,.25)
        for a,b in zip(reversed(rr['coefficients']),baseline['coefficients']):self.assertClose(a,b)
        rr=fit(x,[-v for v in y],.25)
        for a,b in zip(rr['coefficients'],baseline['coefficients']):self.assertClose(a,-b)
        signed=fit([[-row[0],row[1]]for row in x],y,.25)
        self.assertClose(signed['coefficients'][0],-baseline['coefficients'][0])

    def test_row_replication_fixed_mean(self):
        a=fit(X,Y,.25);b=fit(X+X,Y+Y,.25)
        self.assertEqual(a['coefficients'],b['coefficients'])
        self.assertEqual(a['primal'],b['primal'])
        self.assertEqual(a['dual'],b['dual'])

    def test_scale_penalty_equivariance(self):
        # X scaled by c, penalty by c -> beta/c, identical prediction and objective.
        a=fit(X,Y,.25);b=fit([[3*v for v in row]for row in X],Y,.75)
        for aa,bb in zip(a['coefficients'],b['coefficients']):self.assertClose(aa,3*bb)
        self.assertClose(a['primal'],b['primal'])

    def test_no_mutation_rng_or_hidden_standardization(self):
        x=copy.deepcopy(X);y=list(Y);start=[2,-3,1];order=[2,1,0]
        snapshot=copy.deepcopy((x,y,start,order));state=random.getstate()
        fit(x,y,.25,initial=start,order=order);path(x,y,[1,.25]);enumerate_optima(x,y,.25)
        self.assertEqual((x,y,start,order),snapshot);self.assertEqual(random.getstate(),state)
        self.assertEqual(fit([[1],[1]],[2,2],.5)['coefficients'],[1.5])

    def test_soft_threshold_boundary_and_sign(self):
        for v,e in [(-2,-1),(-1,0),(0,0),(1,0),(2,1)]:self.assertEqual(soft(v,1),e)

    def test_input_domains(self):
        bad=[([],[],.25),([[1],[1,2]],[1,2],.25),([[1]],[True],.25),([[True]],[1],.25),([[float('nan')]],[1],.25),([[1]],[101],.25),([[1]],[1],0),([[1]],[1],True),([[1e-6]],[1],.25),([[1]*13],[1],.25)]
        for args in bad:
            with self.subTest(args=args),self.assertRaises(ValueError):fit(*args)
        for kwargs in [{'max_sweeps':True},{'max_sweeps':0},{'tolerance':0},{'order':[0,0,2]},{'order':[False,1,2]},{'initial':[1]}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):fit(X,Y,.25,**kwargs)
        for penalties in [[],[1,1],[.5,1],[True],[0]]:
            with self.assertRaises(ValueError):path(X,Y,penalties)
        with self.assertRaises(ValueError):enumerate_optima([[1]*6],[1],.25)
        with self.assertRaises(ValueError):enumerate_optima([[1]]*33,[1]*33,.25)
        with self.assertRaises(ValueError):witness(X,Y,[1],.25)

    def test_floating_inputs_oracle_exact_binary_semantics(self):
        r=enumerate_optima([[1]],[.3],.1)['witnesses'][0]
        expected=Fraction(.3)-Fraction(.1)
        self.assertEqual(Fraction(r['coefficients'][0]),expected)
        self.assertNotEqual(expected,Fraction(1,5))

    def test_stopping_needs_both_gap_and_kkt(self):
        r=demo()['correlated']['limited']
        self.assertGreater(r['gap'],r['tolerance']);self.assertGreater(r['max_kkt'],r['tolerance'])
        # A tiny coefficient times a nonzero KKT violation gives small gap only.
        r=fit([[1]],[1],.25,initial=[.75+1e-7],tolerance=1e-10)
        self.assertGreater(r['history'][0]['max_kkt'],1e-10)
        self.assertEqual(r['sweeps'],1)

    def test_demo_and_committed_report(self):
        self.assertEqual(demo(),json.loads((ROOT/'examples/demo.json').read_text()))
        self.assertEqual(path(X,Y,[2,1,.5,.25,.125,.0625]),json.loads(subprocess.check_output([sys.executable,'-m','sparselab','path',str(ROOT/'examples/input.json')],cwd=ROOT)))
        self.assertEqual(demo(),json.loads(subprocess.check_output([sys.executable,'-m','sparselab','demo'],cwd=ROOT)))

    def test_cli_strict_schema_utf8_size(self):
        valid=json.loads((ROOT/'examples/input.json').read_text())
        cases=[dict(valid,unknown=1),dict(valid,schema_version=True),dict(valid,schema_version=1.0),dict(valid,penalties=[True])]
        bodies=[json.dumps(v).encode() for v in cases]+[b'{"schema_version":1,"schema_version":1}',b'NaN',b'\xff',b' '*100001]
        with tempfile.TemporaryDirectory()as d:
            for body in bodies:
                p=Path(d)/'input.json';p.write_bytes(body)
                r=subprocess.run([sys.executable,'-m','sparselab','path',str(p)],capture_output=True,cwd=ROOT)
                self.assertEqual(r.returncode,2);self.assertFalse(r.stdout)

    def test_path_propagates_limit_and_actual_warm_start(self):
        r=demo()['correlated']
        result=path(r['X'],r['y'],[.25,.125],max_sweeps=1)
        self.assertFalse(result['all_converged'])
        self.assertTrue(any(not r['converged'] for r in result['results']))
        self.assertEqual(result['results'][1]['initial'],result['results'][0]['coefficients'])
        self.assertTrue(all(len(r['history'])==2 for r in result['results']))

    def test_five_column_oracle_full_enumeration(self):
        x=[[(-1)**((i&j).bit_count()%2) for j in range(1,6)]for i in range(8)]
        target=[1,-.5,.25,0,.125]
        y=[sum(a*b for a,b in zip(row,target)) for row in x]
        exact=enumerate_optima(x,y,.0625)
        self.assertEqual(exact['considered'],243)
        self.assertEqual(exact['singular_sign_patterns'],0)
        self.assertEqual(exact['witness_count'],1)
        expected=[.9375,-.4375,.1875,0,.0625]
        self.assertEqual([float(Fraction(v)) for v in exact['witnesses'][0]['coefficients']],expected)
        self.assertEqual(fit(x,y,.0625)['coefficients'],expected)
