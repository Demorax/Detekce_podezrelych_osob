# MMPose 1.3.2 compatible ViTPose config
default_scope = 'mmpose'

# Model settings
model = dict(
    type='TopdownPoseEstimator',  # Changed from 'TopDown' to 'TopdownPoseEstimator'
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1280,  # Huge model
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
        out_channels=17,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='RegressionLabel',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.)
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True
    )
)

# Dataset settings
dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='Lin, Tsung-Yi and others',
        title='Microsoft COCO: Common objects in context',
        container='ECCV',
        year='2014',
        homepage='http://cocodataset.org/'
    ),
    keypoint_info=dict({
        0: dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1: dict(name='left_eye', id=1, color=[51, 153, 255], type='upper', swap='right_eye'),
        2: dict(name='right_eye', id=2, color=[51, 153, 255], type='upper', swap='left_eye'),
        3: dict(name='left_ear', id=3, color=[51, 153, 255], type='upper', swap='right_ear'),
        4: dict(name='right_ear', id=4, color=[51, 153, 255], type='upper', swap='left_ear'),
        5: dict(name='left_shoulder', id=5, color=[0, 255, 0], type='upper', swap='right_shoulder'),
        6: dict(name='right_shoulder', id=6, color=[255, 128, 0], type='upper', swap='left_shoulder'),
        7: dict(name='left_elbow', id=7, color=[0, 255, 0], type='upper', swap='right_elbow'),
        8: dict(name='right_elbow', id=8, color=[255, 128, 0], type='upper', swap='left_elbow'),
        9: dict(name='left_wrist', id=9, color=[0, 255, 0], type='upper', swap='right_wrist'),
        10: dict(name='right_wrist', id=10, color=[255, 128, 0], type='upper', swap='left_wrist'),
        11: dict(name='left_hip', id=11, color=[0, 255, 0], type='lower', swap='right_hip'),
        12: dict(name='right_hip', id=12, color=[255, 128, 0], type='lower', swap='left_hip'),
        13: dict(name='left_knee', id=13, color=[0, 255, 0], type='lower', swap='right_knee'),
        14: dict(name='right_knee', id=14, color=[255, 128, 0], type='lower', swap='left_knee'),
        15: dict(name='left_ankle', id=15, color=[0, 255, 0], type='lower', swap='right_ankle'),
        16: dict(name='right_ankle', id=16, color=[255, 128, 0], type='lower', swap='left_ankle')
    }),
    skeleton_info=[[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                   [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
                   [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]],
    joint_weights=[1.] * 17,
    sigmas=[0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
            0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
)