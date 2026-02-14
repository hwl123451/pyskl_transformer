# =========================================================
# RGB Only Transformer SOTA 配置
# =========================================================

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ConvAcTransformer',  # 你的 Transformer 类名
        # Transformer 参数
        attention_heads=4,
        num_layers=4,
        num_classes=60,       # NTU-60
        num_frames=8,         # 对应 clip_len=8
        drop_p=0.1,
        feature_extractor_name='wide_resnet50_2',
        learnable_pe=False
    ),
    cls_head=dict(
        type='I3DHead',
        num_classes=60,       # NTU-60
        in_channels=2048,     # WideResNet50 输出
        # [修正] 删除 spatial_type，因为 I3DHead 默认就是 AvgPooling
        # 你的 Backbone 输出已经是 1x1x1，I3DHead 此时做 AvgPool 等于没做，正好兼容
        dropout=0.5,
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    train_cfg=None,
    test_cfg=dict(average_clips='prob')
)

# 使用 PoseDataset 以获得最佳性能（支持骨骼辅助）
dataset_type = 'PoseDataset'
data_root = '/media/zjq/6CDE775EDE771F8E/lxk/NTU-data/nturgbd_videos_all/'
ann_file = '/media/zjq/6CDE775EDE771F8E/lxk/NTU-data/ntu60_hrnet.pkl'

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=1),

    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # [关键] 既然用了 PoseDataset，Flip 必须带 kp 参数
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    # 仅收集 RGB
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

val_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

test_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=10),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=10,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='xsub_train',
        data_prefix=data_root,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='xsub_val',
        data_prefix=data_root,
        pipeline=val_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='xsub_val',
        data_prefix=data_root,
        pipeline=test_pipeline
    )
)

# 优化器配置
# 使用 SGD，学习率 0.01 是比较稳妥的起点
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# [修改前] optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# [修改后] 使用 AdamW，注意学习率要调小！
optimizer = dict(type='AdamW', lr=0.00025, weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

#lr_config = dict(policy='step', step=[12, 16])
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    by_epoch=True,
    warmup='linear',    # 加上 warmup 防止 AdamW 刚开始梯度爆炸
    warmup_iters=5,     # 预热 5 个 Epoch
    warmup_ratio=0.01,
    warmup_by_epoch=True
)
total_epochs = 50
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/rgb_only_transformer_sota_1'
load_from = None
fp16 = dict(loss_scale='dynamic')