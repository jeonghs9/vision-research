"""중복 제거가 합친 사람 박스 쌍을 전부 기록한다.

진짜 두 사람을 합쳤다면 (미탐) 두 박스는 크기가 비슷하고 겹침이 0.75 언저리일 것이다.
같은 사람의 중복이라면 작은 박스가 큰 박스 안에 들어가 겹침이 1.0 에 가깝다.
"""
import sys, csv, argparse
import numpy as np
sys.path.insert(0, 'scripts/04-오토바이다인승')
import detect_motorcycle_multi_rider as M
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='1')
a = ap.parse_args()

model = YOLO('models/det-best.pt')
res = model.track(source=a.input, classes=[0, 2], conf=0.25, iou=0.7, imgsz=1024,
                  device=a.device, tracker='bytetrack.yaml', stream=True,
                  persist=True, verbose=False)

rows = []
for fi, r in enumerate(res):
    pb, pi, pc, mb, mi, mc = M.split_detections(r, 0, 2)
    keep = M.deduplicate_boxes(mb, mc, 0.80) if len(mb) > 1 else list(range(len(mb)))
    mb = [mb[i] for i in keep]; mi = [mi[i] for i in keep]; mc = [mc[i] for i in keep]
    # dedup 을 끈 상태로 연관
    links, _ = M.associate_riders(pb, pi, pc, mb, 0.30, 'person_ioa',
                                  0.08, 6.0, 1.0, True, 0.0)
    for midx, riders in links.items():
        if len(riders) < 2:
            continue
        kept = M.deduplicate_boxes([x.box for x in riders], [x.conf for x in riders], 0.75)
        dropped = [i for i in range(len(riders)) if i not in set(kept)]
        for di in dropped:
            D = riders[di]
            # 어떤 박스에 흡수됐는지 = 겹침이 가장 큰 남은 박스
            best, bs = None, -1
            for ki in kept:
                K = riders[ki]
                inter = M.intersection_area(D.box, K.box)
                s = inter / min(M.box_area(D.box), M.box_area(K.box))
                if s > bs: bs, best = s, K
            aD, aK = M.box_area(D.box), M.box_area(best.box)
            rows.append(dict(
                frame=fi, moto_id=mi[midx],
                kept_id=best.person_id, kept_conf=round(best.conf, 3), kept_area=round(aK),
                drop_id=D.person_id, drop_conf=round(D.conf, 3), drop_area=round(aD),
                inter_min=round(bs, 4),
                area_ratio=round(min(aD, aK) / max(aD, aK), 4),
                kept_box=" ".join(f"{v:.1f}" for v in best.box),
                drop_box=" ".join(f"{v:.1f}" for v in D.box),
            ))

with open(a.out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['frame'])
    w.writeheader(); w.writerows(rows)
print(f"{a.input.split('/')[-1]}  병합 {len(rows)}건")
