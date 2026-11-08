"""Reproducible affine-path graphic; exact report is embedded in full."""
import argparse
import base64
from fractions import Fraction as F
import html
import json
from pathlib import Path
import sys
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sparselab import branch_demo


def render(report):
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="1110" viewBox="0 0 1120 1110"><rect width="1120" height="1110" fill="#f4efec"/><rect x="16" y="16" width="1088" height="1078" rx="22" fill="white"/><g font-family="Arial,sans-serif">']
    def t(x,y,v,size=17,fill='#251f21',weight='400',extra=''):
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" {extra}>{html.escape(str(v))}</text>')
    payload=base64.b64encode(zlib.compress(json.dumps(report,sort_keys=True,separators=(',',':')).encode(),9)).decode()
    out.append('<metadata id="full-branch-report" data-encoding="zlib-base64-json">'+payload+'</metadata>')
    t(48,62,'SparseLab / a feature can leave, then return',30,weight='600')
    t(48,96,'Exact affine LASSO branches, including all sign and inactive-correlation constraints.',18,'#585254')
    t(48,139,'1  Lowering lambda does not imply a growing nonzero support',21,weight='600')
    t(110,180,'Coefficient value',16,'#585254');t(690,180,'Feature 0',17,'#585254');t(850,180,'Feature 1',17,'#317e6b','600')
    for value in [-1,0,1,2,3,4]:
        yy=520-66*value
        out.append(f'<line x1="110" y1="{yy}" x2="1010" y2="{yy}" stroke="#eae9ea"/>')
        t(73,yy+6,value,16,'#585254')
    segments=report['correlated']['segments']
    vertices=[segments[0]['endpoints'][0]]+[r['endpoints'][0]for r in segments[1:]]
    for j,color in enumerate(['#777274','#317e6b']):
        points=' '.join(f'{float(110+(3-F(w["penalty"]))*300):.6f},{float(520-F(w["coefficients"][j])*66):.6f}'for w in vertices)
        out.append(f'<polyline id="coefficient-path-{j}" points="{points}" fill="none" stroke="{color}" stroke-width="4"/>')
        for i,w in enumerate(vertices):
            xx=110+(3-F(w['penalty']))*300;yy=520-F(w['coefficients'][j])*66
            out.append(f'<circle id="vertex-{j}-{i}" cx="{float(xx):.6f}" cy="{float(yy):.6f}" r="5" fill="{color}"/>')
    for i,w in enumerate(vertices):
        xx=float(110+(3-F(w['penalty']))*300)
        t(xx,618,w['penalty'],16,'#585254',extra=f'text-anchor="middle" id="knot-{i}"')
    t(110,651,'lambda decreases from left to right: 3 to 0',16,'#585254')
    t(48,694,'2  Every interval is certified by exact KKT inequalities',21,weight='600')
    t(58,731,'Signs (closure)',16,'#585254');t(300,731,'lambda interval',16,'#585254');t(550,731,'beta 0',16,'#585254');t(810,731,'beta 1',16,'#585254')
    def affine(a,d):
        if d=='0':return a
        if a=='0':return d+' * lambda'
        return a+(' + 'if F(d)>0 else ' - ')+str(abs(F(d)))+' * lambda'
    for i,r in enumerate(segments):
        interval='['+r['lower']+', '+('infinity)'if r['upper']is None else r['upper']+']')
        values=[str(r['signs']),interval]+[affine(a,d)for a,d in zip(r['coefficient_intercepts'],r['coefficient_slopes'])]
        for j,(xx,value)in enumerate(zip([58,300,550,810],values)):
            t(xx,769+39*i,value,17,'#317e6b'if i==3 else '#251f21','600'if i==3 else '400',f'id="branch-cell-{i}-{j}"')
    t(48,967,'Feature 1 leaves at lambda = 1 and re-enters negative at lambda = 1/3.',18,weight='600')
    t(48,1000,'Singular selected Gram: explicit noncertificate. Boundary zeros are allowed, not called active.',17,'#585254')
    t(48,1030,'One-branch verifier, not a general LARS implementation or a causal feature-selection claim.',17,'#585254')
    out.append('<line x1="48" y1="1051" x2="1072" y2="1051" stroke="#eae9ea"/>')
    t(48,1076,'Reproduce: python examples/render_branches.py --check | full exact branch witnesses embedded',15,'#585254')
    out.append('</g></svg>')
    return '\n'.join(out)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    target=Path(__file__).resolve().parents[1]/'assets/regularization-branches.svg';output=render(branch_demo())
    if args.check:
        if target.read_text()!=output:raise SystemExit('branch figure differs from exact report')
    else:target.write_text(output)


if __name__=='__main__':main()
