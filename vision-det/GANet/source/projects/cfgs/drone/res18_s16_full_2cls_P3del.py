"""si_vietnam_drone 2클래스 — GANet-small(ResNet18), P3 단계 제거 버전.

res18_s8_full_2cls.py와 학습 파라미터(해상도·batch·lr·epoch·후처리 칸 단위 값) 전부 동일하고 neck 경로만 다르다.
  기존:  C3 -> C4 -> C5 -> C5+SA -> (C5+SA)+C4 = P4 -> P4+C3 = P3 -> LFA(P3) -> head   (출력 1/8)
  P3del: C3 -> C4 -> C5 -> C5+SA -> (C5+SA)+C4 = P4 -> LFA(P4) -> head              (출력 1/16)
fpn_down_scale 등 파생값을 바꾸므로 model·CollectLanePoints·pipeline·data를 다시 정의한다(상속 시점 값 고정 문제).
"""
from projects.cfgs.drone.res18_s8_full import *  # noqa

dataset_type = 'CulaneDataset2Cls'
num_lane_classes = 2
work_dir = './work_dirs/drone/res18_s16_full_2cls_P3del'

# ---- P3 제거: LFA(DCN)와 head를 P4(1/16)에 건다 ----
fpn_in_channels = [256, 512]   # C4, C5 (backbone은 C3·C4·C5를 내고 neck이 맨 앞 C3를 버림, dcn_fpn.py forward)
fpn_down_scale = [16, 32]
sample_gt_points = [21, 11]    # 기존 P4·P5 값
sample_per_lane = [21, 11]
dcn_point_num = [7, 5]         # P4에 기존 P3의 1x7 LFA
deconv_layer = [True, False]
fpn_layer_num = len(fpn_down_scale)
hm_down_scale = fpn_down_scale[hm_idx]  # 16


model = dict(
    type='GANet',
    backbone=dict(
        type='ResNet',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        depth=18,
        strides=(1, 2, 2, 2),
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='DeformFPN',
        in_channels=fpn_in_channels,
        out_channels=fpn_out_dim,
        trans_idx=fpn_attn_idx,
        trans_cfg=dict(
            attn_in_dims=[fpn_in_channels[fpn_attn_idx], attn_out_dim],
            attn_out_dims=[attn_out_dim, attn_out_dim],
            strides=[1, 1],
            ratios=[4, 4],
            pos_shape=(1, img_scale[1] // 32, img_scale[0] // 32),
        ),
        dcn_point_num=dcn_point_num,
        deconv_layer=deconv_layer,
        deconv_before=deconv_before,
        dcn_only_cls=True,
    ),
    head=dict(
        type='GANetHeadFast',
        num_classes=num_lane_classes,
        in_channels=fpn_out_dim,
        hm_idx=hm_idx,
        kpt_thr=kpt_thr,
        root_thr=root_thr,
        loss_cfg=dict(
            type='LaneLossAggress',
            loss_weights=dict(point=1.0, error=1.0, offset=0.5, aux=0.2),
            use_smooth=False,
            deconv_layer=deconv_layer,
            sample_gt_points=sample_gt_points,
            point_scale=False,
            assigner_cfg=dict(type='LaneAssigner'),
        )
    ),
    post_processing=dict(
        hm_down_scale=hm_down_scale,
        crop_bbox=crop_bbox,
        points_thr=points_thr,
        cluster_by_center_thr=cluster_by_center_thr,
        cluster_thr=cluster_thr,
        group_fast=group_fast,
    )
)

CollectLanePoints = dict(
    type='CollectLanePoints',
    keys=['img', 'targets'],
    max_lane_num=max_lane_num,
    lane_extend=False,
    gaussian_hyper=gaussian_hyper,
    num_classes=num_lane_classes,
    fpn_cfg=dict(
        hm_idx=hm_idx,
        fpn_down_scale=fpn_down_scale,
        sample_per_lane=sample_per_lane,
        deconv_layer=deconv_layer,
    ),
    dis_weights=dis_weights,
)

train_pipeline = [
    dict(type='albumentation', pipelines=train_al_pipeline),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    CollectLanePoints,
]

val_pipeline = [
    dict(type='albumentation', pipelines=val_al_pipeline),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    CollectLanePoints,
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=num_workers,
    train=dict(
        type=dataset_type, data_root=data_root, data_list=train_data_list,
        evaluate_data_list_1=evaluate_data_list_1, evaluate_data_list_s=evaluate_data_list_s,
        pipeline=train_pipeline, test_mode=False, work_dir=work_dir,
    ),
    val=dict(
        type=dataset_type, data_root=data_root, data_list=val_data_list,
        evaluate_data_list_1=evaluate_data_list_1, evaluate_data_list_s=evaluate_data_list_s,
        pipeline=val_pipeline, test_mode=False, work_dir=work_dir,
    ),
    test=dict(
        type=dataset_type, data_root=data_root, data_list=test_data_list,
        evaluate_data_list_1=evaluate_data_list_1, evaluate_data_list_s=evaluate_data_list_s,
        test_suffix='.jpg', pipeline=val_pipeline, test_mode=True, work_dir=work_dir,
    ))
