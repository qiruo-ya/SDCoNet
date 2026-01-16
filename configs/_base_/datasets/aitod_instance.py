dataset_type = 'AITODDataset'
data_root = 'data/AI-TOD/'
# find_unused_parameters = True
# data_root = 'data/AI-TOD/'
backend_args = None
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug',
        scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/aitod_trainval_v1.json',
        data_prefix=dict(img='trainval/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/aitod_test_v1.json',
        data_prefix=dict(img='test/images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/aitod_test_v1.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


# data = dict(
#     # samples_per_gpu=2,
#     # workers_per_gpu=1,
#     train=dict(type=dataset_type,
#                ann_file=data_root + 'annotations/aitod_trainval_v1.json',
#                img_prefix=data_root + 'trainval/images',
#                pipeline=train_pipeline),
#     val=dict(type=dataset_type,
#              ann_file=data_root + 'annotations/aitod_test_v1.json',
#              img_prefix=data_root + 'test/images',
#              pipeline=test_pipeline),
#     test=dict(type=dataset_type,
#               ann_file=data_root + 'annotations/aitod_test_v1.json',
#               img_prefix=data_root + 'test/images',
#               pipeline=test_pipeline)
# )
# evaluation = dict(metric=['bbox'])
