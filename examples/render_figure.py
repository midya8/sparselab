"""Reproducible report figure; visible geometry and full metadata are tested."""
import argparse
import base64
import html
import json
from pathlib import Path
import sys
import zlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparselab import demo


def render(report):
    text=[]
    def t(x,y,value,size=17,fill='#251f21',weight='400',extra=''):
        text.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" {extra}>{html.escape(str(value))}</text>')
    text.append('<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="1040" viewBox="0 0 1120 1040"><rect width="1120" height="1040" fill="#f4efec"/><rect x="16" y="16" width="1088" height="1008" rx="22" fill="white"/><g font-family="Arial,sans-serif">')
    payload=base64.b64encode(zlib.compress(json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),9)).decode()
    text.append('<metadata id="full-report" data-encoding="zlib-base64-json">'+payload+'</metadata>')
    t(48,62,'SparseLab / fitted predictions are not selected identities',29,weight='600')
    t(48,93,'Three developed controls. LASSO objective: squared residual / (2N) + lambda * L1.',18,'#585254')
    t(48,138,'1  Orthogonal path: compare every coefficient with soft-threshold closed forms',20,weight='600')
    # Uniform x on log2(lambda), decreasing lambda left to right.
    rows=report['orthogonal']['path']['results']
    ymap=lambda v: 330 - 74*v
    for value in [-.75,0,1,2]:
        yy=ymap(value);text.append(f'<line x1="108" y1="{yy}" x2="1010" y2="{yy}" stroke="#eae9ea"/>');t(55,yy+5,f'{value:g}',16,'#585254')
    colors=['#317e6b','#6a62a5','#b66b27']
    for j in range(3):
        points=' '.join(f'{124+174*k:.3f},{ymap(r["coefficients"][j]):.3f}'for k,r in enumerate(rows))
        text.append(f'<polyline id="coefficient-{j}" points="{points}" fill="none" stroke="{colors[j]}" stroke-width="3"/>')
        for k,r in enumerate(rows):
            text.append(f'<circle id="point-{j}-{k}" cx="{124+174*k}" cy="{ymap(r["coefficients"][j])}" r="4" fill="{colors[j]}"/>')
        t(700+j*105,168,'beta '+str(j),16,colors[j],weight='600')
    for k,r in enumerate(rows):t(112+174*k,414,f'{r["penalty"]:g}',15,'#585254')
    t(420,443,'lambda (log2 scale, decreasing)',17,'#585254')
    t(48,484,'2  Duplicated feature: different sparse supports, identical predictions',20,weight='600')
    t(48,513,'Columns 0 and 1 are identical. Same y, penalty 1/4 and objective. Only visit order changes.',17,'#585254')
    t(58,548,'Solution',16,'#585254');t(345,548,'beta 0',16,'#585254');t(465,548,'beta 1',16,'#585254');t(585,548,'beta 2',16,'#585254');t(710,548,'Primal',16,'#585254');t(830,548,'max KKT',16,'#585254')
    d=report['duplicate']
    for i,(key,label)in enumerate([('forward','Forward sweep'),('reverse','Reverse sweep'),('midpoint','Convex midpoint')]):
        r=d[key];yy=582+i*38;t(58,yy,label)
        vals=r['coefficients']+[r['primal'],r['max_kkt']]
        for j,(xx,v)in enumerate(zip([345,465,585,710,830],vals)):
            t(xx,yy,f'{v:.6g}',17,'#317e6b','600',f'id="duplicate-{i}-{j}" data-value="{v!r}"')
    t(48,707,'Exact rational oracle finds both sparse witnesses: objective 5/8; predictions agree.',18,weight='600')
    t(48,738,'Every convex mixture of those witnesses is also optimal, including [7/8, 7/8, -1/2].',17,'#585254')
    t(48,782,'3  Correlated columns: a sweep limit must not look like convergence',20,weight='600')
    t(58,817,'Run',16,'#585254');t(345,817,'Sweeps',16,'#585254');t(465,817,'Primal',16,'#585254');t(650,817,'Dual gap',16,'#585254');t(830,817,'max KKT',16,'#585254')
    for i,(key,label)in enumerate([('limited','1-sweep limit'),('converged','Converged fit')]):
        r=report['correlated'][key];yy=852+i*35;t(58,yy,label)
        for j,(xx,v)in enumerate(zip([345,465,650,830],[r['sweeps'],r['primal'],r['gap'],r['max_kkt']])):
            t(xx,yy,f'{v:.6g}',17,'#317e6b' if i else '#585254','600'if i else '400',f'id="convergence-{i}-{j}" data-value="{v!r}"')
    t(48,935,'Exact optimum 19/68. Floating dual/KKT witnesses are not directed-rounding certificates.',17,'#585254')
    t(48,964,'No hidden intercept or scaling. No causal feature selection, held-out accuracy or new algorithm claim.',17,'#585254')
    text.append('<line x1="48" y1="984" x2="1072" y2="984" stroke="#eae9ea"/>')
    t(48,1007,'Reproduce: python examples/render_figure.py --check  |  full report metadata embedded',15,'#585254')
    text.append('</g></svg>')
    return '\n'.join(text)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    destination=Path(__file__).resolve().parents[1]/'assets/sparsity-audit.svg'
    generated=render(demo())
    if args.check:
        if destination.read_text()!=generated:raise SystemExit('figure differs from demo')
    else:destination.write_text(generated)


if __name__=='__main__':main()
