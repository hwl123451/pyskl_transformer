# =========================================================
# RGB Only Transformer 高精度版 (Accuracy First)
# =========================================================

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ConvAcTransformer1',
        frozen_backbone=True,
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
        num_classes=60,
        in_channels=2048,
        # spatial_type=None,  # 你的代码里已经 unsqueeze 了，这里不写或者 None 都可以
        dropout=0.5,          # 保持高 Dropout 防止过拟合
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    train_cfg=None,
    test_cfg=dict(average_clips='prob')
)

dataset_type = 'PoseDataset'
data_root = '/media/zjq/6CDE775EDE771F8E/lxk/NTU-data/nturgbd_videos_all/'
ann_file = '/media/zjq/6CDE775EDE771F8E/lxk/NTU-data/ntu60_hrnet.pkl'

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    # 既然用了 PYSKL 的 Dataset，这里必须是 dict 形式
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
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
    # [关键修改 1] 降 Batch Size 以防 OOM
    videos_per_gpu=16,
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

# [关键修改 2] 梯度累积 (Gradient Accumulation)
# 真实 Batch = 4 (显卡) * 2 (双卡) = 8
# 累积次数 = 4
# 虚拟 Batch = 8 * 4 = 32 (这才是高精度的保证！)
optimizer = dict(type='AdamW', lr=0.0003, weight_decay=0.05) # 虚拟 Batch 32 对应 3e-4 比较稳
optimizer_config = dict(
    grad_clip=dict(max_norm=20, norm_type=2),
    #cumulative_iters=4  # <--- 这就是提分黑科技
)

# [关键修改 3] 100 Epoch + Cosine + Warmup
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    by_epoch=True,
    warmup='linear',
    warmup_iters=5,  # 预热 5 轮
    warmup_ratio=0.01,
    warmup_by_epoch=True
)
total_epochs = 100 # 既然不追求速度，跑满 100 轮才能榨干性能

checkpoint_config = dict(interval=5) # 没必要每轮都存，5轮存一次省硬盘
workflow = [('train', 1)]
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/rgb_only_transformer_accuracy'
load_from = None
fp16 = dict(loss_scale='dynamic') # 3080Ti 必开