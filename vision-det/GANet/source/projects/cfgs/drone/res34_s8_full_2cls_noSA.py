"""si_vietnam_drone 2클래스 — GANet-medium(ResNet34), SA(C5 attention) 제거 버전.

res34_s8_full_2cls.py와 전부 동일, neck의 trans_cfg(SA)만 None.
  noSA: C3 -> C4 -> C5 -> C5+C4 = P4 -> P4+C3 = P3 -> LFA(P3) -> head  (출력 1/8)
SA가 C5를 512ch -> 64ch로 바꾸던 단계가 없어지므로, C5를 받는 lateral conv 입력이 512ch가 된다.
"""
from projects.cfgs.drone.res34_s8_full_2cls import *  # noqa

work_dir = './work_dirs/drone/res34_s8_full_2cls_noSA'
model['neck']['trans_cfg'] = None
for _k in ('train', 'val', 'test'):
    data[_k]['work_dir'] = work_dir
del _k
