"""si_vietnam_drone 2클래스 — GANet-medium(ResNet34), P3 제거 + C5 attention 을 SA 대신 Deformable Attention(DA)으로.

res34_s16_full_2cls_P3del.py와 전부 동일, trans_cfg 의 전역 self-attention 만 DA(단일 스케일, head 8, point 4)로 교체.
pre_conv(512->64) / final_conv / gamma 잔차 / 2층 구조는 SA 와 같음 (dcn_fpn.py DeformAttentionLayer).
pos_mode='query': DETR 방식: 위치 인코딩을 query(어디를 볼지 예측)에만 더함
"""
from projects.cfgs.drone.res34_s16_full_2cls_P3del import *  # noqa

work_dir = './work_dirs/drone/res34_s16_full_2cls_P3del_DAq'
model['neck']['trans_cfg'].update(attn_type='deform', num_heads=8, num_points=4, pos_mode='query')
for _k in ('train', 'val', 'test'):
    data[_k]['work_dir'] = work_dir
del _k
