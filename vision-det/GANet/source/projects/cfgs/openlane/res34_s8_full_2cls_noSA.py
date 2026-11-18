"""[OpenLane용] OpenLane 2클래스 — GANet-medium(ResNet34), P3 사용 + SA 제거.

res34_s8_full_2cls.py와 전부 동일, neck의 trans_cfg(SA)만 None (드론 res34_s8_full_2cls_noSA.py와 같은 변경). C5 lateral conv 입력이 512ch가 된다.
  noSA: C3 -> C4 -> C5 -> C5+C4 = P4 -> P4+C3 = P3 -> LFA(P3) -> head  (출력 1/8)
"""
from projects.cfgs.openlane.res34_s8_full_2cls import *  # noqa

work_dir = './work_dirs/openlane/res34_s8_full_2cls_noSA'
model['neck']['trans_cfg'] = None
for _k in ('train', 'val', 'test'):
    data[_k]['work_dir'] = work_dir
del _k
