# Migrated ViTPose configuration for MMPose 1.x/3.x
# Based on your original config but updated to new format

# Default scope
default_scope = 'mmpose'

# Runtime settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='coco/AP', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
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
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Data configuration
dataset_type = 'CocoDataset'  # Use standard CocoDataset for compatibility
data_mode = 'topdown'
data_root = 'data/coco/'

# Dataset metainfo for CrowdPose (14 keypoints)
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

# Codec configuration
codec = dict(
    type='MSRAHeatmap',
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2
)

# Model configuration - Updated to new format
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
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
    head=dict(
        type='HeatmapHead',
        in_channels=1280,
        out_channels=14,  # CrowdPose has 14 keypoints
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        final_layer=dict(kernel_size=1),
        loss=dict(type='KeypointMSELoss', use_target_weight=True)
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    )
)

# Data pipeline
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# Data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
        metainfo=dataset_info
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dataset_info
    )
)

test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dataset_info
    )
)

# Evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json'
)
test_evaluator = val_evaluator

# Learning rate and optimizer
train_cfg = dict(max_epochs=210, val_interval=10)

optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4))

param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True
    )
]

# Automatically scale LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# Log config
log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True
)