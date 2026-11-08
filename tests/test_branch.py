import copy
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from sparselab import branch, branch_demo, enumerate_optima
ROOT=Path(__file__).resolve().parents[1]


def total(xs):return sum(xs,F(0))


class BranchTests(unittest.TestCase):
    def check_witness(self,r,w):
        x=[[F(v)for v in row]for row in r['X']];y=list(map(F,r['y']));n=len(y);p=len(x[0]);lam=F(w['penalty'])
        a=list(map(F,r['coefficient_intercepts']));d=list(map(F,r['coefficient_slopes']));beta=[u+lam*v for u,v in zip(a,d)]
        self.assertEqual(list(map(F,w['coefficients'])),beta)
        prediction=[total(v*b for v,b in zip(row,beta))for row in x]
        residual=[v-f for v,f in zip(y,prediction)]
        corr=[total(row[j]*v for row,v in zip(x,residual))/n for j in range(p)]
        self.assertEqual(list(map(F,w['prediction'])),prediction);self.assertEqual(list(map(F,w['residual'])),residual)
        self.assertEqual(list(map(F,w['correlations'])),corr)
        self.assertTrue(all(c==(lam if b>0 else -lam)if b else abs(c)<=lam for b,c in zip(beta,corr)))
        u=[v/n for v in residual];self.assertEqual(list(map(F,w['dual_vector'])),u)
        primal=total(v*v for v in residual)/(2*n)+lam*total(abs(v)for v in beta)
        dual=total(v*z for v,z in zip(y,u))-n*total(z*z for z in u)/2
        self.assertEqual(F(w['primal']),primal);self.assertEqual(F(w['dual']),dual);self.assertEqual(primal,dual);self.assertEqual(w['gap'],'0')
        slack=[F(c['intercept'])+lam*F(c['slope'])for c in r['constraints']]
        self.assertEqual(list(map(F,w['slacks'])),slack);self.assertTrue(all(v>=0 for v in slack))
        self.assertEqual(w['tight_constraints'],[i for i,v in enumerate(slack)if v==0])

    def test_demo_intervals_and_independent_closed_form(self):
        rows=branch_demo()['correlated']['segments']
        self.assertEqual([(r['lower'],r['upper'])for r in rows],[('3',None),('4/3','3'),('1','4/3'),('1/3','1'),('0','1/3')])
        self.assertEqual([r['coefficient_intercepts']for r in rows],[['0','0'],['0','3/5'],['4','-1'],['2','0'],['4','-1']])
        self.assertEqual([r['coefficient_slopes']for r in rows],[['0','0'],['0','-1/5'],['-3','1'],['-1','0'],['-7','3']])
        self.assertTrue(all(r['gram']==[['1','2'],['2','5']]and r['cross']==['2','3']for r in rows))

    def test_full_equations_endpoints_and_interior(self):
        demo=branch_demo()
        for r in demo['correlated']['segments']+demo['duplicate_branches']+[demo['singleton']]:
            for w in r['endpoints']+[r['interior']]:self.check_witness(r,w)

    def test_path_continuity_and_drop_reentry(self):
        rows=branch_demo()['correlated']['segments']
        for left,right in zip(rows,rows[1:]):
            self.assertEqual(left['lower'],right['upper'])
            for k in ['coefficients','prediction','primal','dual_vector']:
                self.assertEqual(left['endpoints'][0][k],right['endpoints'][-1][k])
        self.assertEqual(rows[2]['endpoints'][0]['coefficients'],['1','0'])
        self.assertEqual(rows[3]['coefficient_intercepts'][1],'0');self.assertEqual(rows[3]['coefficient_slopes'][1],'0')
        self.assertEqual(rows[4]['endpoints'][0]['coefficients'],['4','-1'])
        self.assertEqual([e['event']for e in branch_demo()['correlated']['events']],['enter_positive','enter_positive','leave','reenter_negative'])

    def test_every_finite_bound_maximal_and_all_limiting_constraints_retained(self):
        for r in branch_demo()['correlated']['segments']:
            lo=F(r['lower']);hi=None if r['upper']is None else F(r['upper'])
            for bound,offset in [(lo,-F(1,100000))]+([]if hi is None else [(hi,F(1,100000))]):
                slacks=[F(c['intercept'])+(bound+offset)*F(c['slope'])for c in r['constraints']]
                self.assertTrue(any(v<0 for v in slacks))
                self.assertTrue(any(F(c['intercept'])+bound*F(c['slope'])==0 for c in r['constraints']))
        zero=branch([[1,1]],[2],[0,0]);self.assertEqual(zero['lower'],'2')
        # Both identical constraints set the same lower knot, not one arbitrary tie.
        tight=zero['endpoints'][0]['tight_constraints'];self.assertEqual(len(tight),2)
        self.assertEqual([zero['constraints'][i]['feature']for i in tight],[0,1])

    def test_seeded_patterns_against_per_lambda_oracle_and_exact_inequalities(self):
        rng=random.Random(829);comparisons=0;certificates=0
        for _ in range(12):
            x=[[rng.randint(-2,2)for _ in range(3)]for _ in range(5)];y=[rng.randint(-3,3)for _ in range(5)]
            for signs in product([-1,0,1],repeat=3):
                r=branch(x,y,signs)
                if not r['certified']:continue
                certificates+=1
                for w in r['endpoints']+[r['interior']]:self.check_witness(r,w)
                lo=F(r['lower']);hi=None if r['upper']is None else F(r['upper'])
                for lam in [F(1,8),F(1,2),F(1),F(2),F(4)]:
                    if lam<lo or (hi is not None and lam>hi):continue
                    beta=[F(a)+lam*F(d)for a,d in zip(r['coefficient_intercepts'],r['coefficient_slopes'])]
                    pred=[total(F(v)*b for v,b in zip(row,beta))for row in x]
                    oracle=enumerate_optima(x,y,float(lam));comparisons+=1
                    self.assertTrue(all(pred==list(map(F,w['prediction']))for w in oracle['witnesses']))
        self.assertGreater(comparisons,50);self.assertGreater(certificates,40)

    def test_singular_is_noncertificate_not_ridge_or_infeasibility(self):
        r=branch_demo()['singular'];self.assertEqual(r['status'],'singular_active_gram');self.assertFalse(r['certified'])
        self.assertNotIn('coefficient_intercepts',r);self.assertNotIn('lower',r)
        for row in branch_demo()['duplicate_branches']:self.assertTrue(row['certified'])
        a,b=branch_demo()['duplicate_branches'];self.assertNotEqual(a['interior']['coefficients'],b['interior']['coefficients'])
        self.assertEqual(a['interior']['prediction'],b['interior']['prediction'])

    def test_infeasible_and_singleton_weak_not_strict_signs(self):
        r=branch_demo()['infeasible'];self.assertFalse(r['certified']);self.assertEqual(r['status'],'infeasible_sign_pattern')
        self.assertTrue(r['impossible_constraints']or F(r['lower'])>F(r['upper']))
        r=branch_demo()['singleton'];self.assertEqual(r['status'],'certified_point');self.assertEqual(r['lower'],'0');self.assertEqual(r['upper'],'0')
        self.assertFalse(r['strict_selected_signs_at_probe']);self.assertEqual(len(r['endpoints']),1)

    def test_zero_columns_zero_response_and_unbounded_empty_support(self):
        r=branch([[0,0]],[1],[0,0]);self.assertTrue(r['certified']);self.assertEqual(r['lower'],'0');self.assertIsNone(r['upper'])
        self.assertEqual(r['interior']['primal'],'1/2');self.assertEqual(r['interior']['gap'],'0')
        self.assertEqual(branch([[0]],[0],[1])['status'],'singular_active_gram')
        r=branch([[1]],[0],[1]);self.assertEqual(r['status'],'certified_point')

    def test_row_replication_permutation_and_column_permutation(self):
        fixture=branch_demo()['correlated'];x=fixture['X'];y=fixture['y'];r=branch(x,y,[1,1])
        for xx,yy in [(x+x,y+y),(x[::-1],y[::-1])]:
            a=branch(xx,yy,[1,1])
            for k in ['gram','cross','lower','upper','coefficient_intercepts','coefficient_slopes','constraints']:self.assertEqual(r[k],a[k])
        a=branch([row[::-1]for row in x],y,[1,1])
        for k in ['coefficient_intercepts','coefficient_slopes']:self.assertEqual(a[k],r[k][::-1])
        self.assertEqual((a['lower'],a['upper']),(r['lower'],r['upper']))

    def test_response_sign_and_column_scaling(self):
        f=branch_demo()['correlated'];x=f['X'];y=f['y'];r=branch(x,y,[1,1])
        neg=branch(x,[-v for v in y],[-1,-1]);self.assertEqual(neg['lower'],r['lower']);self.assertEqual(neg['upper'],r['upper'])
        self.assertEqual(list(map(F,neg['coefficient_intercepts'])),[-F(v)for v in r['coefficient_intercepts']])
        scaled=branch([[2*v for v in row]for row in x],y,[1,1]);self.assertEqual(F(scaled['lower']),2*F(r['lower']));self.assertEqual(F(scaled['upper']),2*F(r['upper']))
        self.assertEqual(list(map(F,scaled['coefficient_intercepts'])),[F(v)/2 for v in r['coefficient_intercepts']])
        self.assertEqual(list(map(F,scaled['coefficient_slopes'])),[F(v)/4 for v in r['coefficient_slopes']])

    def test_binary64_semantics_and_dimension_limits(self):
        r=branch([[1]],[.3],[1]);self.assertEqual(F(r['y'][0]),F(.3));self.assertNotEqual(F(r['y'][0]),F(3,10))
        self.assertEqual(r['upper'],str(F(.3)))
        r=branch([[int(i%12==j)for j in range(12)]for i in range(128)],[1]*128,[1]*12)
        self.assertTrue(r['certified']);self.assertEqual(r['lower'],'0')
        for w in r['endpoints']+[r['interior']]:self.check_witness(r,w)

    def test_bad_inputs_and_no_mutation_or_rng(self):
        x=[[1,2],[2,1]];y=[2,3];signs=[1,0];old=copy.deepcopy((x,y,signs));state=random.getstate();branch(x,y,signs)
        self.assertEqual((x,y,signs),old);self.assertEqual(random.getstate(),state)
        for signs in [[True],[1.0],['1'],[],[2],None]:
            with self.assertRaises(ValueError):branch([[1]],[1],signs)
        for x,y in [([],[]),([[1e-6]],[1]),([[float('inf')]],[1]),([[1]],[False])]:
            with self.assertRaises(ValueError):branch(x,y,[1])

    def test_committed_reports_cli_and_retained_outputs(self):
        self.assertEqual(branch_demo(),json.loads((ROOT/'examples/branch-demo.json').read_text()))
        cases=[(['branch-demo'],branch_demo()),(['branch',str(ROOT/'examples/branch-input.json')],branch_demo()['correlated']['segments'][2])]
        for args,expected in cases:self.assertEqual(json.loads(subprocess.check_output([sys.executable,'-m','sparselab']+args,cwd=ROOT)),expected)
        for mode in ['demo','screen-demo']:
            self.assertEqual(subprocess.check_output([sys.executable,'-m','sparselab',mode],cwd=ROOT),(ROOT/f'examples/{mode}.json').read_bytes())

    def test_strict_cli_rejects_unknown_missing_nonfinite_duplicate_utf8_and_oversize(self):
        original=json.loads((ROOT/'examples/branch-input.json').read_text())
        missing=dict(original);del missing['signs']
        bad=[dict(original,penalty=1),dict(original,schema_version=True),dict(original,signs=[True,0]),missing]
        bodies=[json.dumps(v).encode()for v in bad]+[b'{"schema_version":1,"schema_version":1}',b'NaN',b'\xff',b' '*100001]
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'x.json'
            for body in bodies:
                p.write_bytes(body);r=subprocess.run([sys.executable,'-m','sparselab','branch',str(p)],capture_output=True,cwd=ROOT)
                self.assertEqual(r.returncode,2);self.assertFalse(r.stdout)
