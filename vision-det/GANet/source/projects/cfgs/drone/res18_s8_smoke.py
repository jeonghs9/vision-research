"""si_vietnam_drone(3840x2160, culane_lane_ganet) 용 GANet-small(ResNet18) config.

projects/cfgs/culane/{common,test_common_s8,final_exp_res18_s8}.py를 한 파일로
합치고 우리 데이터에 맞게 고쳤다 (다단계 상속 대신 자체완결 파일로 — 디버깅
편의). 원본 파일은 건드리지 않았다.

스모크 테스트용 설정(작은 batch_size, 짧은 epoch) — 실제 학습 시 조정 필요.
"""
# ---- 데이터 ----
dataset_type = 'CulaneDataset'
data_root = "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/culane_lane_ganet"
train_data_list = [data_root + '/list/train.txt']
val_data_list = [data_root + '/list/val.txt']
test_data_list = [data_root + '/list/test.txt']
evaluate_data_list_1 = val_data_list
evaluate_data_list_s = val_data_list

num_lane_classes = 1  # 실선/점선 구분은 다음 단계 — 지금은 위치만
batch_size = 2
num_workers = 4
work_dir = './work_dirs/drone/res18_s8_smoke'
load_from = None
resume_from = None

# ---- 이미지 크기 ----
# 원본 3840x2160(16:9). CULane처럼 하늘 자르기 대신 전체 프레임을 그대로 쓴다.
# FPN(8,16,32 stride) 전부와 나누어떨어져야 해서 32의 배수로 반올림.
ori_scale = (3840, 2160)
crop_bbox = [0, 0, 3840, 2160]
img_scale = (960, 544)  # (width, height) -- 960/32=30, 544/32=17
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ---- 모델 하이퍼파라미터 (culane/final_exp_res18_s8.py 와 동일) ----
hm_idx = 0
fpn_in_channels = [128, 256, 512]
fpn_out_dim = 64
attn_out_dim = 64
fpn_attn_idx = -1
fpn_down_scale = [8, 16, 32]
sample_gt_points = [41, 21, 11]
sample_per_lane = [41, 21, 11]
dcn_point_num = [7, 5, 3]
deconv_layer = [True, False, False]
fpn_layer_num = len(fpn_down_scale)
hm_down_scale = fpn_down_scale[hm_idx]
deconv_before = False
dis_weights = [1, 0.4, 0.2]
max_lane_num = 13  # 우리 데이터 실측 최대 13개 (culane_lane_ganet 변환 로그 참고)

points_thr = 10
cluster_thr = 3
cluster_by_center_thr = 5
root_thr = 1
kpt_thr = 0.4
gaussian_hyper = dict(radius=2, root_radius=4)
group_fast = True

# pos_shape = (1, H/32, W/32) -- img_scale(960,544) 기준
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

# ---- 데이터 파이프라인 ----
train_compose = dict(bboxes=False, keypoints=True, masks=False)

train_al_pipeline = [
    dict(type='Compose', params=train_compose),
    dict(type='Crop', x_min=crop_bbox[0], x_max=crop_bbox[2],
        y_min=crop_bbox[1], y_max=crop_bbox[3], p=1),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
    dict(type='OneOf', transforms=[
        dict(type='RGBShift', r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
        dict(type='HueSaturationValue', hue_shift_limit=(-10, 10), sat_shift_limit=(-15, 15),
            val_shift_limit=(-10, 10), p=1.0),
    ], p=0.7),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='OneOf', transforms=[
        dict(type='Blur', blur_limit=3, p=1.0),
        dict(type='MedianBlur', blur_limit=3, p=1.0),
    ], p=0.2),
    dict(type='RandomBrightness', limit=0.2, p=0.6),
    dict(type='ShiftScaleRotate', shift_limit=0.1, scale_limit=(-0.2, 0.2), rotate_limit=10,
        border_mode=0, p=0.6),
    dict(type='RandomResizedCrop', height=img_scale[1], width=img_scale[0],
        scale=(0.8, 1.2), ratio=(1.7, 2.7), p=0.6),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
]

val_al_pipeline = [
    dict(type='Compose', params=train_compose),
    dict(type='Crop', x_min=crop_bbox[0], x_max=crop_bbox[2],
        y_min=crop_bbox[1], y_max=crop_bbox[3], p=1),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
]

CollectLanePoints = dict(
    type='CollectLanePoints',
    keys=['img', 'targets'],
    max_lane_num=max_lane_num,
    lane_extend=False,
    gaussian_hyper=gaussian_hyper,
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

# ---- 최적화·런타임 (스모크 테스트용으로 축소) ----
optimizer = dict(type='Adam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='Poly', warmup='linear', warmup_iters=20, warmup_ratio=1.0 / 10, min_lr=1e-5)
workflow = [('train', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=1)  # 스모크: 1epoch만
dist_params = dict(backend='nccl')
find_unused_parameters = True
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1000)  # 스모크 중엔 평가 스킵 (train.py --no-validate 로도 스킵)
log_level = 'INFO'
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
