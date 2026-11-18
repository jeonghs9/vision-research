"""[OpenLane용] OpenLane 2클래스 — GANet-medium(ResNet34), P3 단계 제거(P3del) + SA 사용.

res34_s8_full_2cls.py(OpenLane 기본)와 전부 동일하고 neck 경로만 다르다 (드론 res34_s16_full_2cls_P3del.py와 같은 변경).
  기본:  C3 -> C4 -> C5 -> C5+SA -> (C5+SA)+C4 = P4 -> P4+C3 = P3 -> LFA(P3) -> head   (출력 1/8)
  P3del: C3 -> C4 -> C5 -> C5+SA -> (C5+SA)+C4 = P4 -> LFA(P4) -> head              (출력 1/16)
fpn_down_scale 등 파생값이 model·CollectLanePoints·pipeline·data 안에 박히므로 import 상속 대신 자체완결 파일로 둔다.
"""
# ---- 데이터 ----
dataset_type = 'CulaneDataset2Cls'
data_root = "/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet"
train_data_list = [data_root + '/list/train.txt']
val_data_list = [data_root + '/list/val.txt']
test_data_list = [data_root + '/list/test.txt']
evaluate_data_list_1 = val_data_list
evaluate_data_list_s = val_data_list

num_lane_classes = 2
batch_size = 16  # 드론 res34(1280x736)과 동일. 1280x864는 픽셀 +17% — 메모리는 스모크 테스트에서 실측 (docs/02-openlane/4)
num_workers = 4
work_dir = './work_dirs/openlane/res34_s16_full_2cls_P3del'
load_from = None  # 드론 가중치 사용 안 함
resume_from = None

# ---- 이미지 크기 ----
ori_scale = (1920, 1280)
crop_bbox = [0, 0, 1920, 1280]  # 전체 프레임 (크롭 없음)
img_scale = (1280, 864)  # (width, height) -- 1280/32=40, 864/32=27
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ---- 모델 하이퍼파라미터 (드론 res34_s8_full_2cls 와 동일) ----
hm_idx = 0
fpn_in_channels = [256, 512]   # P3del: C4, C5만 FPN에 (backbone은 C3·C4·C5를 내고 neck이 맨 앞 C3를 버림, dcn_fpn.py forward)
fpn_out_dim = 64
attn_out_dim = 64
fpn_attn_idx = -1
fpn_down_scale = [16, 32]
sample_gt_points = [21, 11]    # 기존 P4·P5 값
sample_per_lane = [21, 11]
dcn_point_num = [7, 5]         # P4에 기존 P3의 1x7 LFA
deconv_layer = [True, False]
fpn_layer_num = len(fpn_down_scale)
hm_down_scale = fpn_down_scale[hm_idx]
deconv_before = False
dis_weights = [1, 0.4, 0.2]
max_lane_num = 16

points_thr = 10
cluster_thr = 3
cluster_by_center_thr = 5
root_thr = 1
kpt_thr = 0.4
gaussian_hyper = dict(radius=2, root_radius=4)
group_fast = True

model = dict(
    type='GANet',
    backbone=dict(
        type='ResNet',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34'),
        depth=34,
        strides=(1, 2, 2, 2),
        num_stages=4,
        out_indices=(1, 2, 3),  # 드론 P3del과 동일 (C3·C4·C5)
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
        scale=(0.8, 1.2), ratio=(1.45, 2.3), p=0.6),
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

# ---- 최적화·런타임 (드론과 동일) ----
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='Poly', warmup='linear', warmup_iters=100, warmup_ratio=1.0 / 10, min_lr=1e-5)
workflow = [('train', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=150)
dist_params = dict(backend='nccl')
find_unused_parameters = True
checkpoint_config = dict(interval=10)
evaluation = dict(interval=100000)  # 사실상 비활성 — --no-validate 로 실행
log_level = 'INFO'
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
