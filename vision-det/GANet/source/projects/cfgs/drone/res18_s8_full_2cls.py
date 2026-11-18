"""si_vietnam_drone 2클래스(실선/점선) 본학습 설정 — GANet-small(ResNet18).

res18_s8_full.py(1클래스, 채택 완료: epoch140)와 해상도·batch·lr·epoch을
전부 동일하게 두고 "클래스 수"만 바꾼 실험이 되도록 설계했다. img_scale 등
파생값이 안 바뀌니 import 상속으로 안전하게 override 가능.

docs/01-model-review/8. 2클래스 구현과 결과.md 참고.
"""
from projects.cfgs.drone.res18_s8_full import *  # noqa

dataset_type = 'CulaneDataset2Cls'
num_lane_classes = 2
work_dir = './work_dirs/drone/res18_s8_full_2cls'

model = dict(
    type='GANet',
    backbone=dict(
        type='ResNet',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        depth=18,
        strides=(1, 2, 2, 2),
        num_stages=4,
        out_indices=(0, 1, 2, 3),
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
