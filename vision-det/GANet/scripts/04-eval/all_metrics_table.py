"""GANet 2클래스 실험 003~013 val(163장) 성능 집계: epoch별 F1 로그를 모아 최고 F1(P/R/오차) + ep100~150 평균/범위."""
import re
import glob
T = '/home/hsjeong/workspace/vision-det/GANet/test'
DOC8 = {  # 문서 8 표 (003·004 epoch별 P, R, F1, 오차)
 '003': [(10,.850,.510,.637,8.16),(20,.885,.587,.706,7.89),(30,.816,.625,.708,7.41),(40,.863,.624,.725,7.61),(50,.871,.627,.729,7.25),(60,.896,.599,.718,7.00),(70,.877,.586,.703,6.72),(80,.913,.562,.696,6.59),(90,.884,.627,.734,6.88),(100,.874,.627,.730,6.77),(110,.921,.597,.724,6.88),(120,.918,.617,.738,6.81),(130,.881,.610,.721,6.63),(140,.906,.615,.732,6.60),(150,.898,.617,.731,6.75)],
 '004': [(10,.863,.568,.685,7.29),(20,.817,.605,.695,8.00),(30,.818,.610,.699,8.08),(40,.884,.531,.663,6.70),(50,.900,.594,.716,6.53),(60,.899,.551,.683,6.22),(70,.861,.625,.724,6.89),(80,.898,.640,.747,6.96),(90,.899,.598,.718,6.62),(100,.888,.628,.736,6.50),(110,.910,.607,.728,6.49),(120,.915,.629,.746,6.62),(130,.888,.648,.749,6.84),(140,.899,.641,.748,6.78),(150,.897,.633,.742,6.61)],
}
PAT = r'precision=([0-9.]+) recall=([0-9.]+) f1=([0-9.]+) avg_match_dist_px=([0-9.]+)'
def from_scorefile(p, tag):
    out = []
    for l in open(p):
        m = re.match(rf'{tag} ep(\d+) RESULT.*' + PAT, l)
        if m: out.append((int(m.group(1)), *map(float, m.groups()[1:])))
    return out
def from_follow(p):
    return [(int(e), *map(float, r)) for e, *r in re.findall(r'=== epoch (\d+).*?' + PAT, open(p).read(), re.S)]
R = dict(DOC8)
R['005'] = from_scorefile(glob.glob(f'{T}/005-*/eval/p3del_score_005.txt')[0], '005')
R['006'] = from_scorefile(glob.glob(f'{T}/006-*/eval/p3del_score_006.txt')[0], '006')
for k in ['007', '008', '009', '010', '011', '012', '013']:
    R[k] = from_follow(glob.glob(f'{T}/{k}-*/eval/{k}.log')[0])
for k, rows in R.items():
    rows = sorted(rows); assert [r[0] for r in rows] == list(range(10, 151, 10)), (k, [r[0] for r in rows])
    b = max(rows, key=lambda r: r[3]); late = [r[3] for r in rows if r[0] >= 100]
    print(f'{k} best ep{b[0]} P{b[1]:.3f} R{b[2]:.3f} F1 {b[3]:.4f} err{b[4]:.2f} | ep100-150 avg {sum(late)/6:.4f} ({min(late):.3f}~{max(late):.3f})')
