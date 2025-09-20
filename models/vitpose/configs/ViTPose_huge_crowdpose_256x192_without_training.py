# Configuration for ViTPose Huge model - Standalone version
# No _base_ imports needed

# Dataset info (from crowdpose.py)
dataset_info = dict(
    dataset_name='crowdpose',
    paper_info=dict(
        author='Li, Jiefeng and Wang, Can and Zhu, Hao and '
        'Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu',
        title='CrowdPose: Efficient Crowded Scenes Pose Estimation '
        'and A New Benchmark',
        container='Proceedings of IEEE Conference on Computer '
        'Vision and Pattern Recognition (CVPR)',
        year='2019',
        homepage='https://github.com/Jeff-sjtu/CrowdPose',
    ),
    keypoint_info={
        0: dict(name='left_shoulder', id=0, color=[51, 153, 255], type='upper', swap='right_shoulder'),
        1: dict(name='right_shoulder', id=1, color=[51, 153, 255], type='upper', swap='left_shoulder'),
        2: dict(name='left_elbow', id=2, color=[51, 153, 255], type='upper', swap='right_elbow'),
        3: dict(name='right_elbow', id=3, color=[51, 153, 255], type='upper', swap='left_elbow'),
        4: dict(name='left_wrist', id=4, color=[51, 153, 255], type='upper', swap='right_wrist'),
        5: dict(name='right_wrist', id=5, color=[0, 255, 0], type='upper', swap='left_wrist'),
        6: dict(name='left_hip', id=6, color=[255, 128, 0], type='lower', swap='right_hip'),
        7: dict(name='right_hip', id=7, color=[0, 255, 0], type='lower', swap='left_hip'),
        8: dict(name='left_knee', id=8, color=[255, 128, 0], type='lower', swap='right_knee'),
        9: dict(name='right_knee', id=9, color=[0, 255, 0], type='lower', swap='left_knee'),
        10: dict(name='left_ankle', id=10, color=[255, 128, 0], type='lower', swap='right_ankle'),
        11: dict(name='right_ankle', id=11, color=[0, 255, 0], type='lower', swap='left_ankle'),
        12: dict(name='top_head', id=12, color=[255, 128, 0], type='upper', swap=''),
        13: dict(name='neck', id=13, color=[0, 255, 0], type='upper', swap='')
    },
    skeleton_info={
        0: dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1: dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2: dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3: dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4: dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5: dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6: dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7: dict(link=('left_shoulder', 'right_shoulder'), id=7, color=[51, 153, 255]),
        8: dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9: dict(link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10: dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11: dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12: dict(link=('top_head', 'neck'), id=12, color=[51, 153, 255]),
        13: dict(link=('right_shoulder', 'neck'), id=13, color=[51, 153, 255]),
        14: dict(link=('left_shoulder', 'neck'), id=14, color=[51, 153, 255])
    },
    joint_weights=[0.2, 0.2, 0.2, 1.3, 1.5, 0.2, 1.3, 1.5, 0.2, 0.2, 0.5, 0.2, 0.2, 0.5],
    sigmas=[0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.079, 0.079]
)

# Evaluation settings
evaluation = dict(interval=10, metric='mAP')

# Optimizer settings
optimizer = dict(type='Adam', lr=5e-4)
optimizer_config = dict(grad_clip=None)

# Learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200]
)
total_epochs = 210

# Channel configuration for CrowdPose (14 keypoints)
channel_cfg = dict(
    num_output_channels=14,
    dataset_joints=14,
    dataset_channel=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
)

# Model settings - ViTPose Huge
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=1280,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11
    )
)

# Data configuration
data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
)

# Data pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(type='TopDownHalfBodyTransform', num_joints_half_body=6, prob_half_body=0.3),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=['image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs']
    ),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs']
    ),
]

test_pipeline = val_pipeline

# Dataset settings
data_root = 'data/crowdpose'
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='TopDownCrowdPoseDataset',
        ann_file=f'{data_root}/annotations/mmpose_crowdpose_trainval.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info=dataset_info
    ),
    val=dict(
        type='TopDownCrowdPoseDataset',
        ann_file=f'{data_root}/annotations/mmpose_crowdpose_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info=dataset_info
    ),
    test=dict(
        type='TopDownCrowdPoseDataset',
        ann_file=f'{data_root}/annotations/mmpose_crowdpose_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info=dataset_info
    )
)

# Runtime settings
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# Environment settings
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Codec settings for newer MMPose versions
codec = dict(
    type='MSRAHeatmap',
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2)

# Default scope
default_scope = 'mmpose'