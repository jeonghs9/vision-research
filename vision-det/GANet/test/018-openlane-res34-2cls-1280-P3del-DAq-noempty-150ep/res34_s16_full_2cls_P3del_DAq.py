dataset_type = 'CulaneDataset2Cls'
data_root = '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet'
train_data_list = [
    '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/train.txt'
]
val_data_list = [
    '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/val.txt'
]
test_data_list = [
    '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/test.txt'
]
evaluate_data_list_1 = [
    '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/val.txt'
]
evaluate_data_list_s = [
    '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/val.txt'
]
num_lane_classes = 2
batch_size = 16
num_workers = 4
work_dir = '/home/hsjeong/workspace/vision-det/GANet/test/018-openlane-res34-2cls-1280-P3del-DAq-noempty-150ep'
load_from = None
resume_from = None
ori_scale = (1920, 1280)
crop_bbox = [0, 0, 1920, 1280]
img_scale = (1280, 864)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
hm_idx = 0
fpn_in_channels = [256, 512]
fpn_out_dim = 64
attn_out_dim = 64
fpn_attn_idx = -1
fpn_down_scale = [16, 32]
sample_gt_points = [21, 11]
sample_per_lane = [21, 11]
dcn_point_num = [7, 5]
deconv_layer = [True, False]
fpn_layer_num = 2
hm_down_scale = 16
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
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='DeformFPN',
        in_channels=[256, 512],
        out_channels=64,
        trans_idx=-1,
        trans_cfg=dict(
            attn_in_dims=[512, 64],
            attn_out_dims=[64, 64],
            strides=[1, 1],
            ratios=[4, 4],
            pos_shape=(1, 27, 40),
            attn_type='deform',
            num_heads=8,
            num_points=4,
            pos_mode='query'),
        dcn_point_num=[7, 5],
        deconv_layer=[True, False],
        deconv_before=False,
        dcn_only_cls=True),
    head=dict(
        type='GANetHeadFast',
        num_classes=2,
        in_channels=64,
        hm_idx=0,
        kpt_thr=0.4,
        root_thr=1,
        loss_cfg=dict(
            type='LaneLossAggress',
            loss_weights=dict(point=1.0, error=1.0, offset=0.5, aux=0.2),
            use_smooth=False,
            deconv_layer=[True, False],
            sample_gt_points=[21, 11],
            point_scale=False,
            assigner_cfg=dict(type='LaneAssigner'))),
    post_processing=dict(
        hm_down_scale=16,
        crop_bbox=[0, 0, 1920, 1280],
        points_thr=10,
        cluster_by_center_thr=5,
        cluster_thr=3,
        group_fast=True))
train_compose = dict(bboxes=False, keypoints=True, masks=False)
train_al_pipeline = [
    dict(
        type='Compose', params=dict(bboxes=False, keypoints=True,
                                    masks=False)),
    dict(type='Crop', x_min=0, x_max=1920, y_min=0, y_max=1280, p=1),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='Resize', height=864, width=1280, p=1),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=(-10, 10),
                sat_shift_limit=(-15, 15),
                val_shift_limit=(-10, 10),
                p=1.0)
        ],
        p=0.7),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.2),
    dict(type='RandomBrightness', limit=0.2, p=0.6),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.1,
        scale_limit=(-0.2, 0.2),
        rotate_limit=10,
        border_mode=0,
        p=0.6),
    dict(
        type='RandomResizedCrop',
        height=864,
        width=1280,
        scale=(0.8, 1.2),
        ratio=(1.45, 2.3),
        p=0.6),
    dict(type='Resize', height=864, width=1280, p=1)
]
val_al_pipeline = [
    dict(
        type='Compose', params=dict(bboxes=False, keypoints=True,
                                    masks=False)),
    dict(type='Crop', x_min=0, x_max=1920, y_min=0, y_max=1280, p=1),
    dict(type='Resize', height=864, width=1280, p=1)
]
CollectLanePoints = dict(
    type='CollectLanePoints',
    keys=['img', 'targets'],
    max_lane_num=16,
    lane_extend=False,
    gaussian_hyper=dict(radius=2, root_radius=4),
    num_classes=2,
    fpn_cfg=dict(
        hm_idx=0,
        fpn_down_scale=[16, 32],
        sample_per_lane=[21, 11],
        deconv_layer=[True, False]),
    dis_weights=[1, 0.4, 0.2])
train_pipeline = [
    dict(
        type='albumentation',
        pipelines=[
            dict(
                type='Compose',
                params=dict(bboxes=False, keypoints=True, masks=False)),
            dict(type='Crop', x_min=0, x_max=1920, y_min=0, y_max=1280, p=1),
            dict(type='HorizontalFlip', p=0.5),
            dict(type='Resize', height=864, width=1280, p=1),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RGBShift',
                        r_shift_limit=10,
                        g_shift_limit=10,
                        b_shift_limit=10,
                        p=1.0),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=(-10, 10),
                        sat_shift_limit=(-15, 15),
                        val_shift_limit=(-10, 10),
                        p=1.0)
                ],
                p=0.7),
            dict(
                type='JpegCompression',
                quality_lower=85,
                quality_upper=95,
                p=0.2),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.2),
            dict(type='RandomBrightness', limit=0.2, p=0.6),
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.1,
                scale_limit=(-0.2, 0.2),
                rotate_limit=10,
                border_mode=0,
                p=0.6),
            dict(
                type='RandomResizedCrop',
                height=864,
                width=1280,
                scale=(0.8, 1.2),
                ratio=(1.45, 2.3),
                p=0.6),
            dict(type='Resize', height=864, width=1280, p=1)
        ]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectLanePoints',
        keys=['img', 'targets'],
        max_lane_num=16,
        lane_extend=False,
        gaussian_hyper=dict(radius=2, root_radius=4),
        num_classes=2,
        fpn_cfg=dict(
            hm_idx=0,
            fpn_down_scale=[16, 32],
            sample_per_lane=[21, 11],
            deconv_layer=[True, False]),
        dis_weights=[1, 0.4, 0.2])
]
val_pipeline = [
    dict(
        type='albumentation',
        pipelines=[
            dict(
                type='Compose',
                params=dict(bboxes=False, keypoints=True, masks=False)),
            dict(type='Crop', x_min=0, x_max=1920, y_min=0, y_max=1280, p=1),
            dict(type='Resize', height=864, width=1280, p=1)
        ]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectLanePoints',
        keys=['img', 'targets'],
        max_lane_num=16,
        lane_extend=False,
        gaussian_hyper=dict(radius=2, root_radius=4),
        num_classes=2,
        fpn_cfg=dict(
            hm_idx=0,
            fpn_down_scale=[16, 32],
            sample_per_lane=[21, 11],
            deconv_layer=[True, False]),
        dis_weights=[1, 0.4, 0.2])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='CulaneDataset2Cls',
        data_root=
        '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet',
        data_list=[
            '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/train.txt'
        ],
        evaluate_data_list_1=[
            '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/val.txt'
        ],
        evaluate_data_list_s=[
            '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/val.txt'
        ],
        pipeline=[
            dict(
                type='albumentation',
                pipelines=[
                    dict(
                        type='Compose',
                        params=dict(bboxes=False, keypoints=True,
                                    masks=False)),
                    dict(
                        type='Crop',
                        x_min=0,
                        x_max=1920,
                        y_min=0,
                        y_max=1280,
                        p=1),
                    dict(type='HorizontalFlip', p=0.5),
                    dict(type='Resize', height=864, width=1280, p=1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RGBShift',
                                r_shift_limit=10,
                                g_shift_limit=10,
                                b_shift_limit=10,
                                p=1.0),
                            dict(
                                type='HueSaturationValue',
                                hue_shift_limit=(-10, 10),
                                sat_shift_limit=(-15, 15),
                                val_shift_limit=(-10, 10),
                                p=1.0)
                        ],
                        p=0.7),
                    dict(
                        type='JpegCompression',
                        quality_lower=85,
                        quality_upper=95,
                        p=0.2),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0)
                        ],
                        p=0.2),
                    dict(type='RandomBrightness', limit=0.2, p=0.6),
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.1,
                        scale_limit=(-0.2, 0.2),
                        rotate_limit=10,
                        border_mode=0,
                        p=0.6),
                    dict(
                        type='RandomResizedCrop',
                        height=864,
                        width=1280,
                        scale=(0.8, 1.2),
                        ratio=(1.45, 2.3),
                        p=0.6),
                    dict(type='Resize', height=864, width=1280, p=1)
                ]),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='CollectLanePoints',
                keys=['img', 'targets'],
                max_lane_num=16,
                lane_extend=False,
                gaussian_hyper=dict(radius=2, root_radius=4),
                num_classes=2,
                fpn_cfg=dict(
                    hm_idx=0,
                    fpn_down_scale=[16, 32],
                    sample_per_lane=[21, 11],
                    deconv_layer=[True, False]),
                dis_weights=[1, 0.4, 0.2])
        ],
        test_mode=False,
        work_dir='./work_dirs/openlane/res34_s16_full_2cls_P3del_DAq'),
    val=dict(
        type='CulaneDataset2Cls',
        data_root=
        '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet',
        data_list=[
            '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/val.txt'
        ],
        evaluate_data_list_1=[
            '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/val.txt'
        ],
        evaluate_data_list_s=[
            '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/val.txt'
        ],
        pipeline=[
            dict(
                type='albumentation',
                pipelines=[
                    dict(
                        type='Compose',
                        params=dict(bboxes=False, keypoints=True,
                                    masks=False)),
                    dict(
                        type='Crop',
                        x_min=0,
                        x_max=1920,
                        y_min=0,
                        y_max=1280,
                        p=1),
                    dict(type='Resize', height=864, width=1280, p=1)
                ]),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='CollectLanePoints',
                keys=['img', 'targets'],
                max_lane_num=16,
                lane_extend=False,
                gaussian_hyper=dict(radius=2, root_radius=4),
                num_classes=2,
                fpn_cfg=dict(
                    hm_idx=0,
                    fpn_down_scale=[16, 32],
                    sample_per_lane=[21, 11],
                    deconv_layer=[True, False]),
                dis_weights=[1, 0.4, 0.2])
        ],
        test_mode=False,
        work_dir='./work_dirs/openlane/res34_s16_full_2cls_P3del_DAq'),
    test=dict(
        type='CulaneDataset2Cls',
        data_root=
        '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet',
        data_list=[
            '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/test.txt'
        ],
        evaluate_data_list_1=[
            '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/val.txt'
        ],
        evaluate_data_list_s=[
            '/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/list/val.txt'
        ],
        test_suffix='.jpg',
        pipeline=[
            dict(
                type='albumentation',
                pipelines=[
                    dict(
                        type='Compose',
                        params=dict(bboxes=False, keypoints=True,
                                    masks=False)),
                    dict(
                        type='Crop',
                        x_min=0,
                        x_max=1920,
                        y_min=0,
                        y_max=1280,
                        p=1),
                    dict(type='Resize', height=864, width=1280, p=1)
                ]),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='CollectLanePoints',
                keys=['img', 'targets'],
                max_lane_num=16,
                lane_extend=False,
                gaussian_hyper=dict(radius=2, root_radius=4),
                num_classes=2,
                fpn_cfg=dict(
                    hm_idx=0,
                    fpn_down_scale=[16, 32],
                    sample_per_lane=[21, 11],
                    deconv_layer=[True, False]),
                dis_weights=[1, 0.4, 0.2])
        ],
        test_mode=True,
        work_dir='./work_dirs/openlane/res34_s16_full_2cls_P3del_DAq'))
optimizer = dict(type='Adam', lr=0.0005, betas=(0.9, 0.999), eps=1e-08)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='Poly',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    min_lr=1e-05)
workflow = [('train', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=150)
dist_params = dict(backend='nccl')
find_unused_parameters = True
checkpoint_config = dict(interval=10)
evaluation = dict(interval=100000)
log_level = 'INFO'
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
auto_resume = False
gpu_ids = [0]
