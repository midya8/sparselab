"""Exact Gap Safe audit figure, with full rational certificate metadata."""
import argparse
import base64
from fractions import Fraction
import html
import json
from pathlib import Path
import sys
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sparselab import screen_demo


def render(report):
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="1080" viewBox="0 0 1120 1080"><rect width="1120" height="1080" fill="#f4efec"/><rect x="16" y="16" width="1088" height="1048" rx="22" fill="white"/><g font-family="Arial,sans-serif">']
    def t(x,y,value,size=17,fill='#251f21',weight='400',extra=''):
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" {extra}>{html.escape(str(value))}</text>')
    payload=base64.b64encode(zlib.compress(json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),9)).decode()
    out.append('<metadata id="full-screen-report" data-encoding="zlib-base64-json">'+payload+'</metadata>')
    t(48,62,'SparseLab / safe zeros need a strict certificate',30,weight='600')
    t(48,94,'Exact rational Gap Safe sphere tests for the stored binary64-input LASSO problem.',18,'#585254')
    t(48,137,'1  A zero coefficient is not the same as a safely screened feature',21,weight='600')
    t(58,181,'Supplied iterate / control',17,'#585254');t(430,181,'Exact gap',17,'#585254');t(635,181,'Radius squared',17,'#585254');t(860,181,'Safe-zero indices',17,'#585254')
    for i,row in enumerate(report['controls']):
        s=row['screen'];yy=220+43*i;t(58,yy,row['name'])
        values=[s['gap'],s['radius_squared'],str(s['safe_zero'])]
        for j,(xx,v)in enumerate(zip([430,635,860],values)):
            t(xx,yy,v,17,'#317e6b'if j==2 and s['safe_zero']else '#251f21','600'if j==2 and s['safe_zero']else '400',f'id="control-{i}-{j}"')
    t(48,501,'Near iterate has beta 2 = 1/32, yet column 2 is certified zero in every optimum.',18,weight='600')
    t(48,532,'At the exact boundary optimum beta 2 = 0, but the strict test cannot certify it.',18,'#585254')
    t(48,575,'2  Square-root-free witness for the near iterate (lambda = 1/4)',21,weight='600')
    t(58,615,'Feature',16,'#585254');t(250,615,'Norm squared * R squared',16,'#585254');t(575,615,'Margin squared',16,'#585254');t(835,615,'Strict slack',16,'#585254')
    near=report['controls'][1]['screen']
    for i,f in enumerate(near['features']):
        yy=652+37*i;t(58,yy,str(i))
        for j,(xx,v)in enumerate(zip([250,575,835],[f['ball_term'],f['margin_squared'],f['strict_slack']])):
            t(xx,yy,v,17,'#317e6b'if i==2 else '#251f21','600'if i==2 else '400',f'id="near-{i}-{j}"')
    # Bars show the same two exact squared quantities for certified column 2.
    f=near['features'][2]
    for i,(key,label,color)in enumerate([('ball_term','Ball term','#b5b0b2'),('margin_squared','Margin squared','#317e6b')]):
        yy=782+i*47;width=float(Fraction(f[key]))*18000
        t(58,yy+21,label,17,'#585254')
        out.append(f'<rect id="bound-bar-{i}" x="250" y="{yy}" width="{width:.6f}" height="26" fill="{color}"/>')
        t(740,yy+21,f[key],17,weight='600'if i else '400')
    t(48,898,'This test certifies zero when margin > 0 and ball_term < margin_squared.',18,weight='600')
    t(48,930,'R squared = 2 * exact_gap / N. Strict equality is not enough; no rounded square root.',17,'#585254')
    t(48,963,'Duplicate features remain uncertified: either copy can carry an optimal nonzero coefficient.',17,'#585254')
    t(48,995,'This audits iterates; no coordinates are dropped from the solver and no speedup is claimed.',17,'#585254')
    out.append('<line x1="48" y1="1020" x2="1072" y2="1020" stroke="#eae9ea"/>')
    t(48,1044,'Reproduce: python examples/render_screening.py --check | full exact witnesses embedded',15,'#585254')
    out.append('</g></svg>')
    return '\n'.join(out)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    target=Path(__file__).resolve().parents[1]/'assets/safe-screening.svg';output=render(screen_demo())
    if args.check:
        if target.read_text()!=output:raise SystemExit('screening figure differs from report')
    else:target.write_text(output)


if __name__=='__main__':main()
