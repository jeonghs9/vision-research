"""[OpenLane용] OpenLane 2클래스 — GANet-medium(ResNet34), P3 사용 + SA 대신 Deformable Attention(DA, pos_mode=query).

res34_s8_full_2cls.py와 전부 동일, C5 attention만 SA -> DA(단일 스케일, head 8, point 4, 위치 인코딩은 query에만)로 교체 (드론 P3del_DAq와 같은 DA 설정, dcn_fpn.py DeformAttentionLayer).
  DAq: C3 -> C4 -> C5 -> C5+DA -> P4 -> P3 -> LFA(P3) -> head  (출력 1/8)
"""
from projects.cfgs.openlane.res34_s8_full_2cls import *  # noqa

work_dir = './work_dirs/openlane/res34_s8_full_2cls_DAq'
model['neck']['trans_cfg'].update(attn_type='deform', num_heads=8, num_points=4, pos_mode='query')
for _k in ('train', 'val', 'test'):
    data[_k]['work_dir'] = work_dir
del _k
