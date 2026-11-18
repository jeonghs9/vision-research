"""[OpenLane용] OpenLane 2클래스 — GANet-medium(ResNet34), P3 제거 + SA 대신 Deformable Attention(DA, pos_mode=query).

res34_s16_full_2cls_P3del.py와 전부 동일, C5 attention만 SA -> DA(단일 스케일, head 8, point 4, 위치 인코딩은 query에만)로 교체 (드론 res34_s16_full_2cls_P3del_DAq.py와 같은 변경).
  P3del-DAq: C3 -> C4 -> C5 -> C5+DA -> P4 -> LFA(P4) -> head  (출력 1/16)
"""
from projects.cfgs.openlane.res34_s16_full_2cls_P3del import *  # noqa

work_dir = './work_dirs/openlane/res34_s16_full_2cls_P3del_DAq'
model['neck']['trans_cfg'].update(attn_type='deform', num_heads=8, num_points=4, pos_mode='query')
for _k in ('train', 'val', 'test'):
    data[_k]['work_dir'] = work_dir
del _k
