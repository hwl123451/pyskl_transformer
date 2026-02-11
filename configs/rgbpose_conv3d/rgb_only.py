model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ConvAcTransformer',
            attention_heads=4,
            num_layers=4,
            num_classes=2,
            num_frames=8,
            drop_p=0.1,
            feature_extractor_name='wide_resnet50_2',
            learnable_pe=False),
    cls_head=dict(
        type='I3DHead',
        in_channels=2048,
        num_classes=2,
        dropout=0.5),
    test_cfg = dict(average_clips='prob'))

# model = dict(
#     type='Recognizer3D',
#     backbone=dict(
#         type='X3D',
#         gamma_w=1.0,            # Width multiplier (1.0 for X3D-S, 2.0 for X3D-M, etc.)
#         gamma_b=2.25,           # Bottleneck expansion
#         gamma_d=2.2,            # Depth multiplier
#         in_channels=3,       # Set to 3 if input is RGB; change if fusing modalities early
#         base_channels=32,       # Default base channels
#         use_swish=True),
#     cls_head=dict(
#         type='I3DHead',
#         in_channels=576,        # For gamma_w=1.0, output channels are typically 432
#         num_classes=60,
#         dropout=0.5),
#     test_cfg=dict(average_clips='prob')
# )

dataset_type = 'PoseDataset'
data_root = '/media/zjq/6CDE775EDE771F8E/lxk/data'# /root/lanyun-fs/data
ann_file = '/media/zjq/6CDE775EDE771F8E/lxk/video_rgbPose/fall_dataset/fall_dataset_skeleton.pkl'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(240, 320), keep_ratio=False),  # <-- use your input size
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(240, 320), keep_ratio=False),  # <-- keep final size as your image shape
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(240, 320), keep_ratio=False),  # <-- use your input size
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    #dict(type='MMUniformSampleFrames', clip_len=dict(RGB=20), num_clips=10),
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(240, 320), keep_ratio=False),  # <-- use your input size
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(type=dataset_type, split='xsub_train', ann_file=ann_file, data_prefix=data_root, pipeline=train_pipeline)),
    val=dict(type=dataset_type, split='xsub_val', ann_file=ann_file, data_prefix=data_root, pipeline=val_pipeline),
    test=dict(type=dataset_type, split='xsub_val', ann_file=ann_file, data_prefix=data_root, pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.15, momentum=0.9, weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 18
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
