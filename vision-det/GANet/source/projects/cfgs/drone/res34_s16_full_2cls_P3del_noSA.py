"""si_vietnam_drone 2클래스 — GANet-medium(ResNet34), P3 단계 + SA(C5 attention) 제거 버전.

res34_s16_full_2cls_P3del.py와 전부 동일, neck의 trans_cfg(SA)만 None.
  P3del-noSA: C3 -> C4 -> C5 -> C5+C4 = P4 -> LFA(P4) -> head  (출력 1/16)
SA가 C5를 512ch -> 64ch로 바꾸던 단계가 없어지므로, C5를 받는 lateral conv 입력이 512ch가 된다.
"""
from projects.cfgs.drone.res34_s16_full_2cls_P3del import *  # noqa

work_dir = './work_dirs/drone/res34_s16_full_2cls_P3del_noSA'
model['neck']['trans_cfg'] = None
for _k in ('train', 'val', 'test'):
    data[_k]['work_dir'] = work_dir
del _k
