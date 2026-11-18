"""[OpenLane용] OpenLane 2클래스 — GANet-medium(ResNet34), P3 제거 + SA 제거.

res34_s16_full_2cls_P3del.py와 전부 동일, neck의 trans_cfg(SA)만 None (드론 res34_s16_full_2cls_P3del_noSA.py와 같은 변경).
  P3del-noSA: C3 -> C4 -> C5 -> C5+C4 = P4 -> LFA(P4) -> head  (출력 1/16)
"""
from projects.cfgs.openlane.res34_s16_full_2cls_P3del import *  # noqa

work_dir = './work_dirs/openlane/res34_s16_full_2cls_P3del_noSA'
model['neck']['trans_cfg'] = None
for _k in ('train', 'val', 'test'):
    data[_k]['work_dir'] = work_dir
del _k
